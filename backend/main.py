"""FastAPI backend for SAM2-powered auto-annotation and YOLO dataset export."""

import io
import os
import re
import secrets
import shutil
import tempfile
import time
import uuid
import zipfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.requests import Request
from fastapi.responses import Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from services.annotation_service import AnnotationService
from services.dataset_service import DatasetService
from services.project_service import ALLOWED_EXTENSIONS, ProjectService
from services.sam2_service import SAM2Service


BASE_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = BASE_DIR / "projects"
MODELS_DIR = BASE_DIR / "models"

project_service = ProjectService(PROJECTS_DIR)
sam2_service = SAM2Service(MODELS_DIR)
annotation_service = AnnotationService(models_dir=MODELS_DIR)
dataset_service = DatasetService(sam2_service)

JOBS: Dict[str, Dict] = {}

MAX_IMAGE_FILE_BYTES = int(os.getenv("MAX_IMAGE_FILE_BYTES", str(20 * 1024 * 1024)))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "1000"))
MAX_ZIP_FILE_BYTES = int(os.getenv("MAX_ZIP_FILE_BYTES", str(1024 * 1024 * 1024)))
MAX_ZIP_MEMBER_BYTES = int(os.getenv("MAX_ZIP_MEMBER_BYTES", str(25 * 1024 * 1024)))
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = int(os.getenv("MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES", str(2 * 1024 * 1024 * 1024)))
MAX_OBJECT_PROMPTS = int(os.getenv("MAX_OBJECT_PROMPTS", "100"))
MAX_OBJECT_PROMPT_CHARS = int(os.getenv("MAX_OBJECT_PROMPT_CHARS", "64"))
MAX_JOB_HISTORY = int(os.getenv("MAX_JOB_HISTORY", "200"))
CLASS_MATCH_MIN_SCORE = float(os.getenv("CLASS_MATCH_MIN_SCORE", "0.68"))
API_KEY = os.getenv("API_KEY", "").strip()
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")

_LABEL_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "and",
    "or",
    "object",
    "objects",
    "item",
    "items",
}

_LABEL_ALIASES = {
    "people": "person",
    "human": "person",
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "bike": "bicycle",
    "cycle": "bicycle",
    "cyclist": "bicycle",
    "automobile": "car",
    "vehicle": "car",
    "sedan": "car",
    "suv": "car",
    "van": "car",
    "cellphone": "phone",
    "smartphone": "phone",
    "mobile": "phone",
    "mobile phone": "phone",
    "cell phone": "phone",
    "telephone": "phone",
    "handset": "phone",
    "kitty": "cat",
    "kitten": "cat",
    "feline": "cat",
    "doggo": "dog",
    "puppy": "dog",
    "canine": "dog",
    "trafficlight": "traffic light",
    "traffic light": "traffic light",
}


def _cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://localhost:5173"]


def _validate_image_bytes(content: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
        return True
    except Exception:
        return False


async def _read_limited_upload(upload: UploadFile, max_bytes: int) -> bytes:
    data: List[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=400, detail=f"File too large. Max allowed: {max_bytes} bytes")
        data.append(chunk)
    return b"".join(data)


async def _save_limited_upload(upload: UploadFile, target_path: Path, max_bytes: int):
    total = 0
    with open(target_path, "wb") as target:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=400, detail=f"File too large. Max allowed: {max_bytes} bytes")
            target.write(chunk)


def _prune_jobs():
    if len(JOBS) <= MAX_JOB_HISTORY:
        return
    removable = [
        job_id
        for job_id, info in JOBS.items()
        if info.get("status") in {"completed", "failed"}
    ]
    overflow = len(JOBS) - MAX_JOB_HISTORY
    for job_id in removable[:overflow]:
        JOBS.pop(job_id, None)


def _safe_project_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Project name cannot be empty")
    if len(cleaned) > 100:
        raise HTTPException(status_code=400, detail="Project name is too long")
    if not re.fullmatch(r"[A-Za-z0-9 _\-.]+", cleaned):
        raise HTTPException(status_code=400, detail="Project name contains unsupported characters")
    return cleaned


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _apply_label_alias(value: str) -> str:
    if not value:
        return value

    direct = _LABEL_ALIASES.get(value)
    if direct:
        return direct

    squashed = value.replace(" ", "")
    return _LABEL_ALIASES.get(squashed, value)


def _singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes", "ses")) and len(token) > 4:
        return token[:-2]
    if token.endswith("sses") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize_label(value: str) -> List[str]:
    normalized = _normalize_label(value)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return []

    normalized = _apply_label_alias(normalized)

    tokens: List[str] = []
    for token in normalized.split(" "):
        if not token or token in _LABEL_STOPWORDS:
            continue
        token = _apply_label_alias(_singularize_token(token))
        if token and token not in _LABEL_STOPWORDS:
            tokens.append(token)
    return tokens


def _canonical_label(value: str) -> str:
    tokens = _tokenize_label(value)
    canonical = " ".join(tokens)
    return _apply_label_alias(canonical)


def _label_match_score(source_label: str, candidate_label: str) -> float:
    if not source_label or not candidate_label:
        return 0.0

    if source_label == candidate_label:
        return 1.0

    source_tokens = set(_tokenize_label(source_label))
    candidate_tokens = set(_tokenize_label(candidate_label))

    token_score = 0.0
    if source_tokens and candidate_tokens:
        overlap = len(source_tokens & candidate_tokens)
        union = len(source_tokens | candidate_tokens)
        token_score = overlap / max(1, union)

    char_score = SequenceMatcher(None, source_label, candidate_label).ratio()
    containment_bonus = 0.0
    if source_label in candidate_label or candidate_label in source_label:
        containment_bonus = 0.2

    semantic_bonus = 0.0
    if len(candidate_tokens) == 1 and next(iter(candidate_tokens)) in source_tokens:
        semantic_bonus = 0.12
    elif len(source_tokens) == 1 and next(iter(source_tokens)) in candidate_tokens:
        semantic_bonus = 0.08

    return min(1.0, (0.65 * token_score) + (0.35 * char_score) + containment_bonus + semantic_bonus)


def _dedupe_objects(objects: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in objects:
        cleaned = (item or "").strip()
        if len(cleaned) > MAX_OBJECT_PROMPT_CHARS:
            cleaned = cleaned[:MAX_OBJECT_PROMPT_CHARS].strip()
        normalized = _canonical_label(cleaned)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(cleaned)
        if len(deduped) >= MAX_OBJECT_PROMPTS:
            break
    return deduped


def _resolve_prompt_objects(project_id: str, requested_objects: List[str]) -> List[str]:
    prompts = _dedupe_objects(requested_objects)
    if prompts:
        return prompts

    classes = project_service.list_classes(project_id)
    class_names = [str(cls.get("name", "")) for cls in classes]
    return _dedupe_objects(class_names)


def _polygon_area(polygon: List[List[float]]) -> float:
    if len(polygon) < 3:
        return 0.0

    area = 0.0
    for idx in range(len(polygon)):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _flat_polygon_to_points(raw_polygon: Any) -> List[List[float]]:
    if not isinstance(raw_polygon, list) or len(raw_polygon) < 6:
        return []

    points: List[List[float]] = []
    for idx in range(0, len(raw_polygon) - 1, 2):
        try:
            x = float(raw_polygon[idx])
            y = float(raw_polygon[idx + 1])
        except (TypeError, ValueError):
            return []
        points.append([x, y])

    return points if len(points) >= 3 else []


def _polygon_from_detection(segmentation: Any, box: Dict[str, Any]) -> List[List[float]]:
    best_polygon: List[List[float]] = []
    best_area = 0.0

    if isinstance(segmentation, list):
        for candidate in segmentation:
            points = _flat_polygon_to_points(candidate)
            if not points:
                continue
            area = _polygon_area(points)
            if area > best_area:
                best_area = area
                best_polygon = points

    if best_polygon:
        return best_polygon

    x1 = float(box.get("x1", 0.0))
    y1 = float(box.get("y1", 0.0))
    x2 = float(box.get("x2", 0.0))
    y2 = float(box.get("y2", 0.0))
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _bbox_from_detection(box: Dict[str, Any], image_width: int, image_height: int) -> Dict[str, float]:
    x1 = max(0.0, min(float(box.get("x1", 0.0)), float(image_width)))
    y1 = max(0.0, min(float(box.get("y1", 0.0)), float(image_height)))
    x2 = max(0.0, min(float(box.get("x2", 0.0)), float(image_width)))
    y2 = max(0.0, min(float(box.get("y2", 0.0)), float(image_height)))

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    safe_w = float(max(1, image_width))
    safe_h = float(max(1, image_height))

    return {
        "x": x1,
        "y": y1,
        "w": w,
        "h": h,
        "cx": x1 + (w / 2.0),
        "cy": y1 + (h / 2.0),
        "x_norm": x1 / safe_w,
        "y_norm": y1 / safe_h,
        "w_norm": w / safe_w,
        "h_norm": h / safe_h,
    }


def _resolve_detected_class_name(
    label: str,
    class_name_map: Dict[str, str],
    prompt_name_map: Optional[Dict[str, str]] = None,
) -> str:
    normalized = _canonical_label(label)
    if not normalized:
        return "unlabeled"

    candidates = dict(class_name_map)
    if prompt_name_map:
        for key, value in prompt_name_map.items():
            candidates.setdefault(key, value)

    if normalized in candidates:
        return candidates[normalized]

    best_name = ""
    best_score = 0.0
    for known_norm, known_name in candidates.items():
        score = _label_match_score(normalized, known_norm)
        if score > best_score:
            best_score = score
            best_name = known_name

    if best_name and best_score >= CLASS_MATCH_MIN_SCORE:
        return best_name

    return label.strip() or "unlabeled"


def _ensure_project_exists(project_id: str):
    try:
        project_service.get_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class AutoAnnotateRequest(BaseModel):
    image_id: Optional[str] = None
    objects: List[str] = Field(default_factory=list)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_box_size: int = Field(default=10, ge=1, le=2048)
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
    history: List[Dict] = Field(default_factory=list)


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
    augmentations: Dict = Field(default_factory=dict)


class AugPreviewRequest(BaseModel):
    image_id: str
    augmentations: Dict = Field(default_factory=dict)


app = FastAPI(
    title="AutoMark SAM2 API",
    description="Production-oriented backend for SAM2 auto-annotation and YOLO export",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        host.strip()
        for host in os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
        if host.strip()
    ]
    + ["testserver"],
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if API_KEY and request.url.path.startswith("/api") and request.url.path != "/api/health":
        provided_key = request.headers.get(API_KEY_HEADER, "")
        if not provided_key or not secrets.compare_digest(provided_key, API_KEY):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Cache-Control"] = "no-store"
    return response


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
        "grounding_dino_loaded": annotation_service._model_loaded,
        "sam_vit_b_loaded": annotation_service._sam_loaded,
    }


@app.get("/api/projects")
async def list_projects():
    projects = project_service.list_projects()
    return {"projects": projects}


@app.post("/api/projects")
async def create_project(payload: ProjectCreateRequest):
    project = project_service.create_project(_safe_project_name(payload.name))
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
    _ensure_project_exists(project_id)

    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files. Max allowed: {MAX_UPLOAD_FILES}")

    with tempfile.TemporaryDirectory() as tmp:
        temp_paths: List[Path] = []
        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                continue
            safe_name = f"{uuid.uuid4()}{suffix}"
            target = Path(tmp) / safe_name
            content = await _read_limited_upload(file, MAX_IMAGE_FILE_BYTES)
            if not _validate_image_bytes(content):
                raise HTTPException(status_code=400, detail=f"Invalid image: {file.filename}")
            target.write_bytes(content)
            temp_paths.append(target)

        added = project_service.add_images_from_paths(project_id, temp_paths)

    return {"added": added, "count": len(added)}


@app.post("/api/projects/{project_id}/upload/zip")
async def upload_zip(project_id: str, file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")

    _ensure_project_exists(project_id)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "images.zip"
        await _save_limited_upload(file, zip_path, MAX_ZIP_FILE_BYTES)

        extracted_paths: List[Path] = []
        total_uncompressed = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            info_items = [item for item in zf.infolist() if not item.is_dir()]
            if len(info_items) > MAX_UPLOAD_FILES:
                raise HTTPException(status_code=400, detail=f"Too many files in ZIP. Max allowed: {MAX_UPLOAD_FILES}")

            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                member_path = Path(member)
                if member_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue

                info = zf.getinfo(member)
                if info.file_size > MAX_ZIP_MEMBER_BYTES:
                    continue
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
                    raise HTTPException(status_code=400, detail="ZIP content too large after extraction")

                target = Path(tmp) / f"{uuid.uuid4()}{member_path.suffix.lower()}"
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if not _validate_image_bytes(target.read_bytes()):
                    target.unlink(missing_ok=True)
                    continue
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
        raise HTTPException(status_code=500, detail="Auto-annotation failed")


async def _run_batch_auto_annotate(project_id: str, image_ids: List[str], req: AutoAnnotateRequest, job_id: str):
    start = time.time()
    processed = 0
    class_name_map: Dict[str, str] = {}
    for cls in project_service.list_classes(project_id):
        raw_name = str(cls.get("name", "")).strip()
        if not raw_name:
            continue
        canonical_name = _canonical_label(raw_name)
        if canonical_name:
            class_name_map[canonical_name] = raw_name

    prompt_name_map: Dict[str, str] = {}
    for name in req.objects:
        canonical_name = _canonical_label(name)
        if canonical_name:
            prompt_name_map[canonical_name] = name

    try:
        warmup_error = await annotation_service.warmup(require_sam=True)
        if warmup_error:
            raise RuntimeError(warmup_error)
    except Exception as exc:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["errors"].append({"image_id": None, "error": "warmup_failed"})
        project_service.log_error(project_id, f"Batch annotate warm-up failed: {exc}")
        return

    for image_id in image_ids:
        item_start = time.time()
        try:
            project_service.set_image_status(project_id, image_id, "annotating")
            image = project_service.get_image(project_id, image_id)
            result = await annotation_service.annotate_image(
                image_path=str(project_service.get_image_path(project_id, image_id)),
                objects=req.objects,
                box_threshold=req.confidence_threshold,
                text_threshold=req.text_threshold,
                use_sam=True,
                nms_threshold=req.nms_threshold,
                min_box_size=req.min_box_size,
            )

            if result.get("error"):
                raise RuntimeError(str(result["error"]))

            image_size = result.get("image_size", {})
            width = int(image_size.get("width") or image.get("width") or 0)
            height = int(image_size.get("height") or image.get("height") or 0)
            if width <= 0 or height <= 0:
                raise RuntimeError("Invalid image dimensions from annotation pipeline")

            boxes = result.get("boxes", [])
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            segmentations = result.get("segmentations", [])
            image_area = float(max(1, width * height))

            masks: List[Dict[str, Any]] = []
            for idx, box in enumerate(boxes):
                polygon = _polygon_from_detection(
                    segmentations[idx] if idx < len(segmentations) else None,
                    box,
                )
                area_ratio = _polygon_area(polygon) / image_area

                if area_ratio < req.min_mask_area_ratio or area_ratio > req.max_mask_area_ratio:
                    continue

                raw_label = str(labels[idx]) if idx < len(labels) else "unlabeled"
                class_name = _resolve_detected_class_name(raw_label, class_name_map, prompt_name_map)
                if class_name != "unlabeled":
                    normalized_class_name = _canonical_label(class_name)
                    if normalized_class_name in class_name_map:
                        class_name = class_name_map[normalized_class_name]
                    else:
                        stored_class = project_service.upsert_class(project_id, class_name)
                        canonical_name = _canonical_label(stored_class["name"])
                        class_name_map[canonical_name] = stored_class["name"]
                        prompt_name_map.setdefault(canonical_name, stored_class["name"])
                        class_name = stored_class["name"]

                masks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "class_name": class_name,
                        "score": float(scores[idx]) if idx < len(scores) else 0.0,
                        "area_ratio": float(area_ratio),
                        "polygon": polygon,
                        "bbox": _bbox_from_detection(box, width, height),
                        "source": "grounding_dino_sam",
                        "visible": True,
                    }
                )

            annotation = {
                "image_id": image_id,
                "filename": image["filename"],
                "width": width,
                "height": height,
                "masks": masks,
                "history": [],
                "updated_at": time.time(),
            }
            project_service.save_annotation(project_id, image_id, annotation)
            JOBS[job_id]["processed"] += 1
            JOBS[job_id]["last_image"] = image["filename"]
        except Exception as exc:
            JOBS[job_id]["skipped"] += 1
            JOBS[job_id]["errors"].append({"image_id": image_id, "error": "annotation_failed"})
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

    if JOBS[job_id]["status"] == "running":
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

    objects = _resolve_prompt_objects(project_id, payload.objects)
    if not objects:
        raise HTTPException(
            status_code=400,
            detail="objects must contain at least one prompt or the project must already define classes",
        )
    payload.objects = objects

    _prune_jobs()

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
        "objects": objects,
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
        project_service.log_error(project_id, f"Augmentation preview failed: {exc}")
        raise HTTPException(status_code=500, detail="Augmentation preview failed")


@app.post("/api/projects/{project_id}/export")
async def export_dataset(project_id: str, payload: ExportRequest):
    if payload.task not in {"segment", "detect"}:
        raise HTTPException(status_code=400, detail="task must be 'segment' or 'detect'")

    try:
        project = project_service.get_project(project_id)
        classes = project_service.list_classes(project_id)
        annotations = project_service.all_annotations(project_id)
        project_dir = project_service.project_dir(project_id)

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
        raise HTTPException(status_code=500, detail="Export failed")


@app.get("/api/projects/{project_id}/export/download")
async def download_export(project_id: str):
    try:
        zip_path = project_service.project_dir(project_id) / "exports" / "dataset_export.zip"
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Export ZIP not found")
    return FileResponse(zip_path, filename=f"{project_id}_dataset_export.zip", media_type="application/zip")


@app.get("/api/projects/{project_id}/export/coco")
async def download_coco_json(project_id: str):
    try:
        coco_path = project_service.project_dir(project_id) / "exports" / "dataset" / "coco_annotations.json"
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not coco_path.exists():
        raise HTTPException(status_code=404, detail="COCO JSON not found")
    return FileResponse(coco_path, filename=f"{project_id}_coco_annotations.json", media_type="application/json")


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    try:
        project_dir = project_service.project_dir(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    shutil.rmtree(project_dir)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
