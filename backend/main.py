"""FastAPI backend for SAM2-powered auto-annotation and YOLO dataset export."""

import io
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from services.dataset_service import DatasetService
from services.project_service import ALLOWED_EXTENSIONS, ProjectService
from services.sam2_service import SAM2Service


BASE_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = BASE_DIR / "projects"
MODELS_DIR = BASE_DIR / "models"

project_service = ProjectService(PROJECTS_DIR)
sam2_service = SAM2Service(MODELS_DIR)
dataset_service = DatasetService(sam2_service)

JOBS: Dict[str, Dict] = {}


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class AutoAnnotateRequest(BaseModel):
    image_id: Optional[str] = None
    confidence_threshold: float = Field(default=0.5, ge=0.5, le=1.0)
    min_mask_area_ratio: float = Field(default=0.001, ge=0.0, le=1.0)
    max_mask_area_ratio: float = Field(default=0.95, ge=0.0, le=1.0)
    auto_class: bool = False


class PointPromptRequest(BaseModel):
    image_id: str
    x: float
    y: float


class BoxPromptRequest(BaseModel):
    image_id: str
    x1: float
    y1: float
    x2: float
    y2: float


class AnnotationUpdateRequest(BaseModel):
    masks: List[Dict]
    history: List[Dict] = []


class ClassUpsertRequest(BaseModel):
    name: str
    color: Optional[str] = None


class ClassRenameRequest(BaseModel):
    class_id: str
    new_name: str


class ClassMergeRequest(BaseModel):
    source_class_id: str
    target_class_id: str


class ShortcutAssignRequest(BaseModel):
    class_id: str
    shortcut: int = Field(ge=1, le=9)


class ExportRequest(BaseModel):
    task: str = Field(default="segment")
    val_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    augmentations: Dict = {}


class AugPreviewRequest(BaseModel):
    image_id: str
    augmentations: Dict = {}


app = FastAPI(
    title="AutoMark SAM2 API",
    description="Production-oriented backend for SAM2 auto-annotation and YOLO export",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "AutoMark SAM2 API",
        "status": "running",
        "projects_dir": str(PROJECTS_DIR),
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": bool(cv2.cuda.getCudaEnabledDeviceCount() > 0) if hasattr(cv2, "cuda") else False,
        "sam2_loaded": sam2_service.loaded,
        "sam2_error": sam2_service.load_error,
        "sam2_device": sam2_service.runtime.device,
    }


@app.get("/api/projects")
async def list_projects():
    projects = project_service.list_projects()
    return {"projects": projects}


@app.post("/api/projects")
async def create_project(payload: ProjectCreateRequest):
    project = project_service.create_project(payload.name)
    return project


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    try:
        project = project_service.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    images = project_service.list_images(project_id)
    classes = project_service.list_classes(project_id)
    return {"project": project, "images": images, "classes": classes}


@app.post("/api/projects/{project_id}/upload/images")
async def upload_images(project_id: str, files: List[UploadFile] = File(...)):
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    with tempfile.TemporaryDirectory() as tmp:
        temp_paths: List[Path] = []
        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                continue
            safe_name = f"{uuid.uuid4()}{suffix}"
            target = Path(tmp) / safe_name
            content = await file.read()
            target.write_bytes(content)
            temp_paths.append(target)

        added = project_service.add_images_from_paths(project_id, temp_paths)

    return {"added": added, "count": len(added)}


@app.post("/api/projects/{project_id}/upload/zip")
async def upload_zip(project_id: str, file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")

    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "images.zip"
        zip_path.write_bytes(await file.read())

        extracted_paths: List[Path] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                member_path = Path(member)
                if member_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue

                target = Path(tmp) / f"{uuid.uuid4()}{member_path.suffix.lower()}"
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_paths.append(target)

        added = project_service.add_images_from_paths(project_id, extracted_paths)

    return {"added": added, "count": len(added)}


@app.get("/api/projects/{project_id}/images")
async def list_images(project_id: str):
    try:
        images = project_service.list_images(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"images": images}


@app.get("/api/projects/{project_id}/images/{image_id}")
async def serve_image(project_id: str, image_id: str):
    try:
        image_path = project_service.get_image_path(project_id, image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file missing")
    return FileResponse(image_path)


@app.get("/api/projects/{project_id}/annotations/{image_id}")
async def get_annotation(project_id: str, image_id: str):
    try:
        annotation = project_service.load_annotation(project_id, image_id)
        return annotation
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.put("/api/projects/{project_id}/annotations/{image_id}")
async def update_annotation(project_id: str, image_id: str, payload: AnnotationUpdateRequest):
    try:
        image = project_service.get_image(project_id, image_id)
        annotation = {
            "image_id": image_id,
            "filename": image["filename"],
            "width": image["width"],
            "height": image["height"],
            "masks": payload.masks,
            "history": payload.history,
            "updated_at": time.time(),
        }
        project_service.save_annotation(project_id, image_id, annotation)
        return annotation
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/api/projects/{project_id}/annotations/{image_id}/masks/{mask_id}")
async def delete_mask(project_id: str, image_id: str, mask_id: str):
    try:
        annotation = project_service.load_annotation(project_id, image_id)
        annotation["masks"] = [mask for mask in annotation.get("masks", []) if mask.get("id") != mask_id]
        project_service.save_annotation(project_id, image_id, annotation)
        return {"ok": True, "mask_count": len(annotation["masks"])}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/projects/{project_id}/annotate/auto")
async def auto_annotate_single(project_id: str, payload: AutoAnnotateRequest):
    if not payload.image_id:
        raise HTTPException(status_code=400, detail="image_id is required")

    try:
        image = project_service.get_image(project_id, payload.image_id)
        project_service.set_image_status(project_id, payload.image_id, "annotating")
        result = sam2_service.auto_annotate(
            image_path=project_service.get_image_path(project_id, payload.image_id),
            confidence_threshold=payload.confidence_threshold,
            min_mask_area_ratio=payload.min_mask_area_ratio,
            max_mask_area_ratio=payload.max_mask_area_ratio,
        )

        masks = result["masks"]
        if payload.auto_class:
            for mask in masks:
                suggested = sam2_service.suggest_class(mask)
                mask["class_name"] = suggested
                project_service.upsert_class(project_id, suggested)

        annotation = {
            "image_id": payload.image_id,
            "filename": image["filename"],
            "width": result["width"],
            "height": result["height"],
            "masks": masks,
            "history": [],
            "updated_at": time.time(),
        }
        project_service.save_annotation(project_id, payload.image_id, annotation)
        return annotation
    except Exception as exc:
        project_service.log_error(project_id, f"Auto-annotate failed for {payload.image_id}: {exc}")
        project_service.set_image_status(project_id, payload.image_id, "unannotated")
        raise HTTPException(status_code=500, detail=str(exc))


async def _run_batch_auto_annotate(project_id: str, image_ids: List[str], req: AutoAnnotateRequest, job_id: str):
    start = time.time()
    processed = 0

    for image_id in image_ids:
        item_start = time.time()
        try:
            project_service.set_image_status(project_id, image_id, "annotating")
            image = project_service.get_image(project_id, image_id)
            result = sam2_service.auto_annotate(
                image_path=project_service.get_image_path(project_id, image_id),
                confidence_threshold=req.confidence_threshold,
                min_mask_area_ratio=req.min_mask_area_ratio,
                max_mask_area_ratio=req.max_mask_area_ratio,
            )

            masks = result["masks"]
            if req.auto_class:
                for mask in masks:
                    suggested = sam2_service.suggest_class(mask)
                    mask["class_name"] = suggested
                    project_service.upsert_class(project_id, suggested)

            annotation = {
                "image_id": image_id,
                "filename": image["filename"],
                "width": result["width"],
                "height": result["height"],
                "masks": masks,
                "history": [],
                "updated_at": time.time(),
            }
            project_service.save_annotation(project_id, image_id, annotation)
            JOBS[job_id]["processed"] += 1
            JOBS[job_id]["last_image"] = image["filename"]
        except Exception as exc:
            JOBS[job_id]["skipped"] += 1
            JOBS[job_id]["errors"].append({"image_id": image_id, "error": str(exc)})
            project_service.log_error(project_id, f"Batch annotate failed for {image_id}: {exc}")
            project_service.set_image_status(project_id, image_id, "unannotated")

        processed += 1
        elapsed = time.time() - start
        avg = elapsed / processed if processed else 0
        remaining = max(0, len(image_ids) - processed)
        JOBS[job_id]["progress"] = round(100 * processed / len(image_ids), 2)
        JOBS[job_id]["eta_seconds"] = int(avg * remaining)
        JOBS[job_id]["elapsed_seconds"] = int(elapsed)
        JOBS[job_id]["last_image_duration_ms"] = int((time.time() - item_start) * 1000)

    JOBS[job_id]["status"] = "completed"


@app.post("/api/projects/{project_id}/annotate/auto/batch")
async def auto_annotate_batch(project_id: str, payload: AutoAnnotateRequest, background_tasks: BackgroundTasks):
    try:
        images = project_service.list_images(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    image_ids = [item["id"] for item in images]
    if not image_ids:
        raise HTTPException(status_code=400, detail="No images uploaded")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "id": job_id,
        "project_id": project_id,
        "status": "running",
        "processed": 0,
        "skipped": 0,
        "total": len(image_ids),
        "progress": 0.0,
        "eta_seconds": None,
        "elapsed_seconds": 0,
        "last_image": None,
        "last_image_duration_ms": None,
        "errors": [],
    }

    background_tasks.add_task(_run_batch_auto_annotate, project_id, image_ids, payload, job_id)
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


@app.post("/api/projects/{project_id}/annotate/point")
async def annotate_by_point(project_id: str, payload: PointPromptRequest):
    try:
        annotation = project_service.load_annotation(project_id, payload.image_id)
        mask = sam2_service.prompt_by_point(
            image_path=project_service.get_image_path(project_id, payload.image_id),
            x=payload.x,
            y=payload.y,
        )
        if mask is None:
            raise HTTPException(status_code=404, detail="No mask found for that point")

        annotation.setdefault("masks", []).append(mask)
        annotation["updated_at"] = time.time()
        project_service.save_annotation(project_id, payload.image_id, annotation)
        return annotation
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/projects/{project_id}/annotate/box")
async def annotate_by_box(project_id: str, payload: BoxPromptRequest):
    try:
        annotation = project_service.load_annotation(project_id, payload.image_id)
        mask = sam2_service.prompt_by_box(
            image_path=project_service.get_image_path(project_id, payload.image_id),
            box=[payload.x1, payload.y1, payload.x2, payload.y2],
        )
        if mask is None:
            raise HTTPException(status_code=404, detail="No mask found for that box")

        annotation.setdefault("masks", []).append(mask)
        annotation["updated_at"] = time.time()
        project_service.save_annotation(project_id, payload.image_id, annotation)
        return annotation
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/projects/{project_id}/classes")
async def list_classes(project_id: str):
    try:
        classes = project_service.list_classes(project_id)
        return {"classes": classes}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/projects/{project_id}/classes")
async def create_or_update_class(project_id: str, payload: ClassUpsertRequest):
    cls = project_service.upsert_class(project_id, payload.name, payload.color)
    return cls


@app.post("/api/projects/{project_id}/classes/shortcut")
async def assign_shortcut(project_id: str, payload: ShortcutAssignRequest):
    try:
        project_service.assign_shortcut(project_id, payload.class_id, payload.shortcut)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}


@app.post("/api/projects/{project_id}/classes/rename")
async def rename_class(project_id: str, payload: ClassRenameRequest):
    try:
        project_service.rename_class(project_id, payload.class_id, payload.new_name)
        return {"ok": True}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/projects/{project_id}/classes/merge")
async def merge_classes(project_id: str, payload: ClassMergeRequest):
    try:
        project_service.merge_classes(project_id, payload.source_class_id, payload.target_class_id)
        return {"ok": True}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/api/projects/{project_id}/classes/{class_id}")
async def delete_class(project_id: str, class_id: str):
    project_service.delete_class(project_id, class_id)
    return {"ok": True}


@app.get("/api/projects/{project_id}/stats")
async def dataset_stats(project_id: str):
    try:
        images = project_service.list_images(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    annotations = project_service.all_annotations(project_id)
    classes = project_service.list_classes(project_id)

    annotated = 0
    total_objects = 0
    per_class: Dict[str, int] = {}

    for image in images:
        ann = annotations.get(image["id"], {})
        masks = ann.get("masks", [])
        labeled = [m for m in masks if m.get("class_name") and m.get("class_name") != "unlabeled"]
        if labeled:
            annotated += 1
        total_objects += len(labeled)
        for mask in labeled:
            cls = mask["class_name"]
            per_class[cls] = per_class.get(cls, 0) + 1

    warnings: List[str] = []
    for cls, count in per_class.items():
        if count < 50:
            warnings.append(f"Class '{cls}' has only {count} examples - add at least 50")

    if per_class:
        sorted_pairs = sorted(per_class.items(), key=lambda kv: kv[1], reverse=True)
        largest = sorted_pairs[0]
        smallest = sorted_pairs[-1]
        if smallest[1] > 0 and largest[1] / smallest[1] >= 3.0:
            warnings.append(
                f"Class imbalance detected - {largest[0]}({largest[1]}) vs {smallest[0]}({smallest[1]})"
            )

    return {
        "total_images": len(images),
        "annotated_images": annotated,
        "remaining_images": len(images) - annotated,
        "average_objects_per_image": round(total_objects / max(1, len(images)), 2),
        "per_class": per_class,
        "known_classes": classes,
        "warnings": warnings,
    }


@app.post("/api/projects/{project_id}/augment/preview")
async def augmentation_preview(project_id: str, payload: AugPreviewRequest):
    try:
        annotation = project_service.load_annotation(project_id, payload.image_id)
        image_path = project_service.get_image_path(project_id, payload.image_id)
        img = dataset_service.preview_augmentation(image_path, annotation.get("masks", []), payload.augmentations)
        ok, encoded = cv2.imencode(".jpg", img)
        if not ok:
            raise ValueError("Failed to encode preview image")
        return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/projects/{project_id}/export")
async def export_dataset(project_id: str, payload: ExportRequest):
    if payload.task not in {"segment", "detect"}:
        raise HTTPException(status_code=400, detail="task must be 'segment' or 'detect'")

    try:
        project = project_service.get_project(project_id)
        classes = project_service.list_classes(project_id)
        annotations = project_service.all_annotations(project_id)
        project_dir = PROJECTS_DIR / project_id

        result = dataset_service.export_dataset(
            project_dir=project_dir,
            project_data=project,
            annotations_by_image_id=annotations,
            classes=classes,
            export_task=payload.task,
            val_ratio=payload.val_ratio,
            augmentations=payload.augmentations,
        )

        return {
            "download_url": f"/api/projects/{project_id}/export/download",
            "stats": result["stats"],
        }
    except Exception as exc:
        project_service.log_error(project_id, f"Export failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}/export/download")
async def download_export(project_id: str):
    zip_path = PROJECTS_DIR / project_id / "exports" / "dataset_export.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Export ZIP not found")
    return FileResponse(zip_path, filename=f"{project_id}_dataset_export.zip", media_type="application/zip")


@app.get("/api/projects/{project_id}/export/coco")
async def download_coco_json(project_id: str):
    coco_path = PROJECTS_DIR / project_id / "exports" / "dataset" / "coco_annotations.json"
    if not coco_path.exists():
        raise HTTPException(status_code=404, detail="COCO JSON not found")
    return FileResponse(coco_path, filename=f"{project_id}_coco_annotations.json", media_type="application/json")


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    shutil.rmtree(project_dir)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
