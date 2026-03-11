import json
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from services.sam2_service import SAM2Service


@dataclass
class AugmentationOptions:
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotations: Optional[List[int]] = None
    random_rotate_small: bool = False
    brightness_contrast: bool = False
    mosaic: bool = False
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None


class DatasetService:
    def __init__(self, sam2_service: SAM2Service):
        self.sam2 = sam2_service

    def export_dataset(
        self,
        project_dir: Path,
        project_data: Dict,
        annotations_by_image_id: Dict[str, Dict],
        classes: List[Dict],
        export_task: str = "segment",
        val_ratio: float = 0.2,
        augmentations: Optional[Dict] = None,
    ) -> Dict:
        export_root = project_dir / "exports"
        export_root.mkdir(parents=True, exist_ok=True)
        dataset_dir = export_root / "dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        classes_sorted = sorted(classes, key=lambda c: c["name"].lower())
        class_to_id = {cls["name"]: idx for idx, cls in enumerate(classes_sorted)}

        base_samples = self._build_base_samples(project_dir, project_data, annotations_by_image_id)
        if not base_samples:
            raise ValueError("No annotated images available for export")

        aug_opts = AugmentationOptions(**(augmentations or {}))
        all_samples = []
        for sample in base_samples:
            all_samples.append(sample)
            all_samples.extend(self._create_augmented_samples(sample, base_samples, aug_opts))

        random.shuffle(all_samples)
        split_idx = max(1, int(len(all_samples) * (1.0 - val_ratio)))
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:] if split_idx < len(all_samples) else []

        stats = {
            "total_images": len(all_samples),
            "total_annotations": 0,
            "per_class": {cls["name"]: 0 for cls in classes_sorted},
            "train_images": len(train_samples),
            "val_images": len(val_samples),
        }

        self._write_split(dataset_dir, "train", train_samples, class_to_id, export_task, stats)
        self._write_split(dataset_dir, "val", val_samples, class_to_id, export_task, stats)

        data_yaml = {
            "path": "./dataset",
            "train": "images/train",
            "val": "images/val",
            "nc": len(classes_sorted),
            "names": [cls["name"] for cls in classes_sorted],
        }

        with open(dataset_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)

        coco_json = self._build_coco(project_data, all_samples, class_to_id)
        with open(dataset_dir / "coco_annotations.json", "w", encoding="utf-8") as f:
            json.dump(coco_json, f, indent=2)

        with open(dataset_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        zip_path = export_root / "dataset_export.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in dataset_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(export_root))

        return {
            "dataset_dir": str(dataset_dir),
            "zip_path": str(zip_path),
            "stats": stats,
        }

    def preview_augmentation(self, image_path: Path, masks: List[Dict], augmentations: Dict) -> np.ndarray:
        sample = {
            "image": cv2.imread(str(image_path)),
            "masks": masks,
            "image_id": "preview",
            "filename": image_path.name,
            "width": 0,
            "height": 0,
        }
        sample["height"], sample["width"] = sample["image"].shape[:2]
        opts = AugmentationOptions(**augmentations)
        generated = self._create_augmented_samples(sample, [sample], opts)
        if not generated:
            return sample["image"]
        return generated[0]["image"]

    def _build_base_samples(self, project_dir: Path, project_data: Dict, annotations_by_image_id: Dict[str, Dict]) -> List[Dict]:
        samples: List[Dict] = []
        for image in project_data.get("images", []):
            ann = annotations_by_image_id.get(image["id"], {})
            masks = [m for m in ann.get("masks", []) if len(m.get("polygon", [])) >= 3 and m.get("class_name") and m.get("class_name") != "unlabeled"]
            if not masks:
                continue
            image_path = project_dir / image["rel_path"]
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            samples.append(
                {
                    "image": img,
                    "masks": masks,
                    "image_id": image["id"],
                    "filename": image["filename"],
                    "width": image["width"],
                    "height": image["height"],
                }
            )
        return samples

    def _create_augmented_samples(self, sample: Dict, all_samples: List[Dict], opts: AugmentationOptions) -> List[Dict]:
        results: List[Dict] = []

        if opts.horizontal_flip:
            results.append(self._flip(sample, horizontal=True))
        if opts.vertical_flip:
            results.append(self._flip(sample, horizontal=False))

        for deg in opts.rotations or []:
            if deg in {90, 180, 270}:
                results.append(self._rotate_right_angle(sample, deg))

        if opts.random_rotate_small:
            deg = random.uniform(-15.0, 15.0)
            results.append(self._rotate_arbitrary(sample, deg))

        if opts.brightness_contrast:
            results.append(self._brightness_contrast(sample))

        if opts.mosaic and len(all_samples) >= 4:
            picked = random.sample(all_samples, 4)
            results.append(self._mosaic(picked))

        if opts.resize_width and opts.resize_height:
            resized = []
            for s in results:
                resized.append(self._resize(s, int(opts.resize_width), int(opts.resize_height)))
            results = resized

        normalized: List[Dict] = []
        for idx, aug in enumerate(results):
            aug["filename"] = f"{Path(sample['filename']).stem}_aug_{idx}.jpg"
            normalized.append(aug)
        return normalized

    def _write_split(self, dataset_dir: Path, split: str, samples: List[Dict], class_to_id: Dict[str, int], export_task: str, stats: Dict):
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        for i, sample in enumerate(samples):
            filename = sample["filename"]
            out_name = f"{Path(filename).stem}_{i}.jpg"
            label_name = f"{Path(out_name).stem}.txt"

            img_path = images_dir / out_name
            label_path = labels_dir / label_name
            cv2.imwrite(str(img_path), sample["image"])

            h, w = sample["image"].shape[:2]
            lines = []
            for mask in sample["masks"]:
                class_name = mask.get("class_name")
                if class_name not in class_to_id:
                    continue
                class_id = class_to_id[class_name]
                polygon = mask.get("polygon", [])
                if len(polygon) < 3:
                    continue

                if export_task == "detect":
                    line = self.sam2.polygon_to_yolo_bbox(polygon, class_id, w, h)
                else:
                    line = self.sam2.polygon_to_yolo_segment(polygon, class_id, w, h)

                if line:
                    lines.append(line)
                    stats["total_annotations"] += 1
                    stats["per_class"][class_name] = stats["per_class"].get(class_name, 0) + 1

            label_path.write_text("\n".join(lines), encoding="utf-8")

    def _build_coco(self, project_data: Dict, samples: List[Dict], class_to_id: Dict[str, int]) -> Dict:
        categories = [{"id": v + 1, "name": k} for k, v in class_to_id.items()]
        images = []
        annotations = []
        ann_id = 1

        for idx, sample in enumerate(samples, start=1):
            h, w = sample["image"].shape[:2]
            images.append({"id": idx, "file_name": sample["filename"], "width": w, "height": h})

            for mask in sample["masks"]:
                class_name = mask.get("class_name")
                if class_name not in class_to_id:
                    continue
                polygon = mask.get("polygon", [])
                if len(polygon) < 3:
                    continue

                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = self._polygon_area(polygon)
                segmentation = [self._flatten_polygon(polygon)]

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": idx,
                        "category_id": class_to_id[class_name] + 1,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": segmentation,
                    }
                )
                ann_id += 1

        return {
            "info": {
                "description": project_data.get("name", "Auto-Annotation Dataset"),
                "version": "1.0",
            },
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    @staticmethod
    def _polygon_area(polygon: List[List[float]]) -> float:
        if len(polygon) < 3:
            return 0.0
        area = 0.0
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    @staticmethod
    def _flatten_polygon(polygon: List[List[float]]) -> List[float]:
        flattened: List[float] = []
        for x, y in polygon:
            flattened.extend([float(x), float(y)])
        return flattened

    def _flip(self, sample: Dict, horizontal: bool) -> Dict:
        image = sample["image"]
        h, w = image.shape[:2]
        flipped = cv2.flip(image, 1 if horizontal else 0)

        masks = []
        for mask in sample["masks"]:
            poly = []
            for x, y in mask["polygon"]:
                nx = (w - 1 - x) if horizontal else x
                ny = y if horizontal else (h - 1 - y)
                poly.append([float(nx), float(ny)])
            masks.append({**mask, "polygon": poly})

        return {**sample, "image": flipped, "masks": masks, "width": w, "height": h}

    def _rotate_right_angle(self, sample: Dict, deg: int) -> Dict:
        image = sample["image"]
        h, w = image.shape[:2]

        if deg == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            new_w, new_h = h, w

            def transform(x, y):
                return h - 1 - y, x
        elif deg == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
            new_w, new_h = w, h

            def transform(x, y):
                return w - 1 - x, h - 1 - y
        else:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_w, new_h = h, w

            def transform(x, y):
                return y, w - 1 - x

        masks = []
        for mask in sample["masks"]:
            poly = []
            for x, y in mask["polygon"]:
                nx, ny = transform(x, y)
                poly.append([float(nx), float(ny)])
            masks.append({**mask, "polygon": poly})

        return {**sample, "image": rotated, "masks": masks, "width": new_w, "height": new_h}

    def _rotate_arbitrary(self, sample: Dict, deg: float) -> Dict:
        image = sample["image"]
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, deg, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        masks = []
        for mask in sample["masks"]:
            poly = []
            for x, y in mask["polygon"]:
                px = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
                py = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
                px = float(min(max(px, 0.0), w - 1.0))
                py = float(min(max(py, 0.0), h - 1.0))
                poly.append([px, py])
            masks.append({**mask, "polygon": poly})

        return {**sample, "image": rotated, "masks": masks, "width": w, "height": h}

    def _brightness_contrast(self, sample: Dict) -> Dict:
        image = sample["image"]
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-25.0, 25.0)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return {**sample, "image": adjusted}

    def _resize(self, sample: Dict, target_w: int, target_h: int) -> Dict:
        image = sample["image"]
        h, w = image.shape[:2]
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        sx = target_w / float(w)
        sy = target_h / float(h)
        masks = []
        for mask in sample["masks"]:
            poly = [[float(x * sx), float(y * sy)] for x, y in mask["polygon"]]
            masks.append({**mask, "polygon": poly})

        return {**sample, "image": resized, "masks": masks, "width": target_w, "height": target_h}

    def _mosaic(self, samples: List[Dict]) -> Dict:
        base_h, base_w = samples[0]["image"].shape[:2]
        out_h, out_w = base_h * 2, base_w * 2
        mosaic = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        offsets = [(0, 0), (0, base_w), (base_h, 0), (base_h, base_w)]
        all_masks = []

        for sample, (oy, ox) in zip(samples, offsets):
            img = cv2.resize(sample["image"], (base_w, base_h))
            mosaic[oy : oy + base_h, ox : ox + base_w] = img
            h, w = sample["image"].shape[:2]
            sx, sy = base_w / float(w), base_h / float(h)
            for mask in sample["masks"]:
                poly = []
                for x, y in mask["polygon"]:
                    poly.append([float(x * sx + ox), float(y * sy + oy)])
                all_masks.append({**mask, "polygon": poly})

        return {
            "image": mosaic,
            "masks": all_masks,
            "image_id": "mosaic",
            "filename": "mosaic.jpg",
            "width": out_w,
            "height": out_h,
        }
