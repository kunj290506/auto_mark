import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageRecord:
    id: str
    filename: str
    rel_path: str
    width: int
    height: int
    status: str = "unannotated"


class ProjectService:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_project_id(project_id: str) -> str:
        try:
            return str(uuid.UUID(project_id))
        except Exception as exc:
            raise FileNotFoundError(f"Project {project_id} not found") from exc

    def _project_dir(self, project_id: str) -> Path:
        normalized_project_id = self._normalize_project_id(project_id)
        candidate = (self.base_dir / normalized_project_id).resolve()
        try:
            candidate.relative_to(self.base_dir)
        except ValueError as exc:
            raise FileNotFoundError(f"Project {project_id} not found") from exc
        return candidate

    def project_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id)

    def _project_file(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    def _classes_file(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "classes.json"

    def _images_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "images"

    def _annotations_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "annotations"

    def _exports_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "exports"

    def _logs_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "logs"

    def _annotation_file(self, project_id: str, image_id: str) -> Path:
        return self._annotations_dir(project_id) / f"{image_id}.json"

    def create_project(self, name: str) -> Dict:
        project_id = str(uuid.uuid4())
        project_dir = self._project_dir(project_id)
        (project_dir / "images").mkdir(parents=True, exist_ok=True)
        (project_dir / "annotations").mkdir(parents=True, exist_ok=True)
        (project_dir / "exports").mkdir(parents=True, exist_ok=True)
        (project_dir / "logs").mkdir(parents=True, exist_ok=True)

        project_data = {
            "id": project_id,
            "name": name,
            "created_at": self._now_iso(),
            "updated_at": self._now_iso(),
            "images": [],
        }
        self._write_json(self._project_file(project_id), project_data)
        self._write_json(self._classes_file(project_id), {"classes": []})
        return project_data

    def list_projects(self) -> List[Dict]:
        projects: List[Dict] = []
        for project_dir in self.base_dir.iterdir():
            if not project_dir.is_dir():
                continue
            project_file = project_dir / "project.json"
            if project_file.exists():
                projects.append(self._read_json(project_file))
        return sorted(projects, key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_project(self, project_id: str) -> Dict:
        project_file = self._project_file(project_id)
        if not project_file.exists():
            raise FileNotFoundError(f"Project {project_id} not found")
        return self._read_json(project_file)

    def save_project(self, project_data: Dict):
        project_data["updated_at"] = self._now_iso()
        self._write_json(self._project_file(project_data["id"]), project_data)

    def add_images_from_paths(self, project_id: str, local_paths: List[Path]) -> List[Dict]:
        project = self.get_project(project_id)
        existing_names = {item["filename"] for item in project.get("images", [])}
        images_dir = self._images_dir(project_id)
        images_dir.mkdir(parents=True, exist_ok=True)

        added: List[Dict] = []
        for src in local_paths:
            if src.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            filename = self._safe_unique_name(src.name, existing_names)
            existing_names.add(filename)
            target = images_dir / filename
            shutil.copy2(src, target)

            width, height = self._get_image_size(target)
            image_id = str(uuid.uuid4())
            record = ImageRecord(
                id=image_id,
                filename=filename,
                rel_path=f"images/{filename}",
                width=width,
                height=height,
                status="unannotated",
            )
            record_dict = record.__dict__
            project["images"].append(record_dict)
            added.append(record_dict)

        self.save_project(project)
        return added

    def list_images(self, project_id: str) -> List[Dict]:
        project = self.get_project(project_id)
        return project.get("images", [])

    def get_image(self, project_id: str, image_id: str) -> Dict:
        project = self.get_project(project_id)
        for item in project.get("images", []):
            if item["id"] == image_id:
                return item
        raise FileNotFoundError(f"Image {image_id} not found")

    def get_image_path(self, project_id: str, image_id: str) -> Path:
        image = self.get_image(project_id, image_id)
        return self._project_dir(project_id) / image["rel_path"]

    def list_classes(self, project_id: str) -> List[Dict]:
        data = self._read_json(self._classes_file(project_id))
        return data.get("classes", [])

    def save_classes(self, project_id: str, classes: List[Dict]):
        self._write_json(self._classes_file(project_id), {"classes": classes})

    def upsert_class(self, project_id: str, class_name: str, color: Optional[str] = None) -> Dict:
        classes = self.list_classes(project_id)
        for cls in classes:
            if cls["name"].lower() == class_name.lower():
                if color:
                    cls["color"] = color
                self.save_classes(project_id, classes)
                return cls

        new_cls = {
            "id": str(uuid.uuid4()),
            "name": class_name,
            "color": color or self._deterministic_color(class_name),
            "shortcut": None,
        }
        classes.append(new_cls)
        self.save_classes(project_id, classes)
        return new_cls

    def assign_shortcut(self, project_id: str, class_id: str, shortcut: int):
        classes = self.list_classes(project_id)
        for cls in classes:
            if cls["id"] == class_id:
                cls["shortcut"] = shortcut
        self.save_classes(project_id, classes)

    def rename_class(self, project_id: str, class_id: str, new_name: str):
        classes = self.list_classes(project_id)
        old_name = None
        for cls in classes:
            if cls["id"] == class_id:
                old_name = cls["name"]
                cls["name"] = new_name
                break
        if old_name is None:
            raise FileNotFoundError("Class not found")
        self.save_classes(project_id, classes)
        self._replace_class_name_in_annotations(project_id, old_name, new_name)

    def merge_classes(self, project_id: str, source_class_id: str, target_class_id: str):
        classes = self.list_classes(project_id)
        source_name = None
        target_name = None
        kept: List[Dict] = []
        for cls in classes:
            if cls["id"] == source_class_id:
                source_name = cls["name"]
                continue
            if cls["id"] == target_class_id:
                target_name = cls["name"]
            kept.append(cls)

        if source_name is None or target_name is None:
            raise FileNotFoundError("Class not found")

        self.save_classes(project_id, kept)
        self._replace_class_name_in_annotations(project_id, source_name, target_name)

    def delete_class(self, project_id: str, class_id: str):
        classes = self.list_classes(project_id)
        removed_name = None
        kept: List[Dict] = []
        for cls in classes:
            if cls["id"] == class_id:
                removed_name = cls["name"]
                continue
            kept.append(cls)
        self.save_classes(project_id, kept)

        if removed_name:
            annotations_dir = self._annotations_dir(project_id)
            for ann_file in annotations_dir.glob("*.json"):
                ann = self._read_json(ann_file)
                changed = False
                for mask in ann.get("masks", []):
                    if mask.get("class_name") == removed_name:
                        mask["class_name"] = "unlabeled"
                        changed = True
                if changed:
                    self._write_json(ann_file, ann)

    def save_annotation(self, project_id: str, image_id: str, annotation: Dict):
        ann_file = self._annotation_file(project_id, image_id)
        self._write_json(ann_file, annotation)
        self._set_image_status(project_id, image_id, "done" if annotation.get("masks") else "unannotated")

    def load_annotation(self, project_id: str, image_id: str) -> Dict:
        ann_file = self._annotation_file(project_id, image_id)
        if ann_file.exists():
            return self._read_json(ann_file)
        image = self.get_image(project_id, image_id)
        return {
            "image_id": image_id,
            "filename": image["filename"],
            "width": image["width"],
            "height": image["height"],
            "masks": [],
            "history": [],
            "updated_at": self._now_iso(),
        }

    def all_annotations(self, project_id: str) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        project = self.get_project(project_id)
        for image in project.get("images", []):
            result[image["id"]] = self.load_annotation(project_id, image["id"])
        return result

    def log_error(self, project_id: str, message: str):
        log_file = self._logs_dir(project_id) / "errors.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{self._now_iso()}] {message}\n")

    def export_dir(self, project_id: str) -> Path:
        path = self._exports_dir(project_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _replace_class_name_in_annotations(self, project_id: str, source: str, target: str):
        annotations_dir = self._annotations_dir(project_id)
        for ann_file in annotations_dir.glob("*.json"):
            ann = self._read_json(ann_file)
            changed = False
            for mask in ann.get("masks", []):
                if mask.get("class_name") == source:
                    mask["class_name"] = target
                    changed = True
            if changed:
                self._write_json(ann_file, ann)

    def _set_image_status(self, project_id: str, image_id: str, status: str):
        project = self.get_project(project_id)
        for image in project.get("images", []):
            if image["id"] == image_id:
                image["status"] = status
                break
        self.save_project(project)

    def set_image_status(self, project_id: str, image_id: str, status: str):
        self._set_image_status(project_id, image_id, status)

    @staticmethod
    def _safe_unique_name(filename: str, used: set) -> str:
        stem = Path(filename).stem
        ext = Path(filename).suffix
        candidate = filename
        i = 1
        while candidate in used:
            candidate = f"{stem}_{i}{ext}"
            i += 1
        return candidate

    @staticmethod
    def _get_image_size(path: Path):
        with Image.open(path) as img:
            return img.width, img.height

    @staticmethod
    def _write_json(path: Path, payload: Dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _read_json(path: Path) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _deterministic_color(key: str) -> str:
        palette = [
            "#ef4444", "#f97316", "#f59e0b", "#84cc16", "#22c55e", "#14b8a6", "#06b6d4",
            "#3b82f6", "#6366f1", "#8b5cf6", "#d946ef", "#ec4899", "#64748b"
        ]
        idx = abs(hash(key)) % len(palette)
        return palette[idx]
