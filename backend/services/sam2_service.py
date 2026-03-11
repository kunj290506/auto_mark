import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


@dataclass
class Sam2Config:
    config_file: str
    checkpoint_path: str
    device: str


class SAM2Service:
    def __init__(self, models_dir: Path, config_file: str = "sam2_hiera_large.yaml", checkpoint_name: str = "sam2_hiera_large.pt"):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = Sam2Config(
            config_file=config_file,
            checkpoint_path=str(self.models_dir / checkpoint_name),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model = None
        self.mask_generator = None
        self.predictor = None
        self.loaded = False
        self.load_error: Optional[str] = None

    def ensure_loaded(self):
        if self.loaded:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            self.model = build_sam2(
                config_file=self.runtime.config_file,
                ckpt_path=self.runtime.checkpoint_path,
                device=self.runtime.device,
            )
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.model,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )

            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                self.predictor = SAM2ImagePredictor(self.model)
            except Exception:
                self.predictor = None

            self.loaded = True
            self.load_error = None
        except Exception as exc:
            self.load_error = str(exc)
            raise RuntimeError(f"Failed to load SAM2: {exc}")

    def auto_annotate(
        self,
        image_path: Path,
        min_mask_area_ratio: float = 0.001,
        max_mask_area_ratio: float = 0.95,
        confidence_threshold: float = 0.5,
        epsilon: float = 2.0,
    ) -> Dict:
        self.ensure_loaded()

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Unable to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        total_pixels = float(width * height)

        masks = self.mask_generator.generate(image_rgb)
        records: List[Dict] = []

        for idx, item in enumerate(masks):
            score = float(item.get("predicted_iou", 0.0))
            if score < confidence_threshold:
                continue

            segmentation = item.get("segmentation")
            if segmentation is None:
                continue

            mask_arr = segmentation.astype(np.uint8)
            area = float(np.count_nonzero(mask_arr))
            area_ratio = area / total_pixels
            if area_ratio < min_mask_area_ratio or area_ratio > max_mask_area_ratio:
                continue

            polygon, bbox = self.mask_to_polygon(mask_arr, width, height, epsilon=epsilon)
            if polygon is None:
                continue

            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "class_name": "unlabeled",
                    "score": score,
                    "area_ratio": area_ratio,
                    "polygon": polygon,
                    "bbox": bbox,
                    "source": "auto",
                    "visible": True,
                }
            )

        self.clear_cuda_cache()
        return {
            "width": width,
            "height": height,
            "masks": records,
        }

    def prompt_by_point(self, image_path: Path, x: float, y: float, epsilon: float = 2.0) -> Optional[Dict]:
        self.ensure_loaded()
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Unable to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        if self.predictor is None:
            return self._fallback_prompt_point(image_rgb, x, y, epsilon)

        self.predictor.set_image(image_rgb)
        coords = np.array([[x, y]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            return None

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(np.uint8)
        polygon, bbox = self.mask_to_polygon(best_mask, width, height, epsilon)
        if polygon is None:
            return None

        return {
            "id": str(uuid.uuid4()),
            "class_name": "unlabeled",
            "score": float(scores[best_idx]),
            "area_ratio": float(np.count_nonzero(best_mask)) / float(width * height),
            "polygon": polygon,
            "bbox": bbox,
            "source": "point_prompt",
            "visible": True,
        }

    def prompt_by_box(self, image_path: Path, box: List[float], epsilon: float = 2.0) -> Optional[Dict]:
        self.ensure_loaded()
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Unable to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        x1, y1, x2, y2 = box
        if self.predictor is None:
            return self._fallback_prompt_box(image_rgb, [x1, y1, x2, y2], epsilon)

        self.predictor.set_image(image_rgb)
        box_arr = np.array([x1, y1, x2, y2], dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            box=box_arr,
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            return None

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(np.uint8)
        polygon, bbox = self.mask_to_polygon(best_mask, width, height, epsilon)
        if polygon is None:
            return None

        return {
            "id": str(uuid.uuid4()),
            "class_name": "unlabeled",
            "score": float(scores[best_idx]),
            "area_ratio": float(np.count_nonzero(best_mask)) / float(width * height),
            "polygon": polygon,
            "bbox": bbox,
            "source": "box_prompt",
            "visible": True,
        }

    @staticmethod
    def mask_to_polygon(mask: np.ndarray, img_w: int, img_h: int, epsilon: float = 2.0) -> Tuple[Optional[List[List[float]]], Optional[Dict]]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) < 3:
            return None, None

        pts = approx.reshape(-1, 2).astype(float)
        polygon = [[float(x), float(y)] for x, y in pts]

        x, y, w, h = cv2.boundingRect(largest)
        bbox = {
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "cx": float(x + w / 2.0),
            "cy": float(y + h / 2.0),
            "x_norm": float(x / img_w),
            "y_norm": float(y / img_h),
            "w_norm": float(w / img_w),
            "h_norm": float(h / img_h),
        }
        return polygon, bbox

    @staticmethod
    def polygon_to_yolo_bbox(polygon: List[List[float]], class_id: int, img_w: int, img_h: int) -> str:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cx = ((x_min + x_max) / 2.0) / img_w
        cy = ((y_min + y_max) / 2.0) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    @staticmethod
    def polygon_to_yolo_segment(polygon: List[List[float]], class_id: int, img_w: int, img_h: int) -> Optional[str]:
        if len(polygon) < 3:
            return None
        normalized = []
        for x, y in polygon:
            normalized.extend([x / img_w, y / img_h])
        values = " ".join(f"{v:.6f}" for v in normalized)
        return f"{class_id} {values}"

    @staticmethod
    def suggest_class(mask: Dict) -> str:
        bbox = mask.get("bbox", {})
        w = float(bbox.get("w", 0.0))
        h = float(bbox.get("h", 0.0))
        if h <= 0 or w <= 0:
            return "object"

        ratio = max(w, h) / max(1.0, min(w, h))
        area_ratio = float(mask.get("area_ratio", 0.0))
        if ratio < 1.3 and area_ratio < 0.05:
            return "ball"
        if ratio > 2.2:
            return "bar"
        if area_ratio > 0.25:
            return "large_object"
        return "box"

    @staticmethod
    def clear_cuda_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _fallback_prompt_point(self, image_rgb: np.ndarray, x: float, y: float, epsilon: float) -> Optional[Dict]:
        auto = self.mask_generator.generate(image_rgb)
        best = None
        best_score = -math.inf
        for item in auto:
            mask = item.get("segmentation")
            if mask is None:
                continue
            h, w = mask.shape[:2]
            xi = int(max(0, min(w - 1, round(x))))
            yi = int(max(0, min(h - 1, round(y))))
            if mask[yi, xi]:
                score = float(item.get("predicted_iou", 0.0))
                if score > best_score:
                    best = item
                    best_score = score

        if best is None:
            return None

        h, w = image_rgb.shape[:2]
        polygon, bbox = self.mask_to_polygon(best["segmentation"].astype(np.uint8), w, h, epsilon)
        if polygon is None:
            return None

        return {
            "id": str(uuid.uuid4()),
            "class_name": "unlabeled",
            "score": float(best.get("predicted_iou", 0.0)),
            "area_ratio": float(np.count_nonzero(best["segmentation"])) / float(w * h),
            "polygon": polygon,
            "bbox": bbox,
            "source": "point_prompt",
            "visible": True,
        }

    def _fallback_prompt_box(self, image_rgb: np.ndarray, box: List[float], epsilon: float) -> Optional[Dict]:
        auto = self.mask_generator.generate(image_rgb)
        x1, y1, x2, y2 = box
        query = np.array([x1, y1, x2, y2], dtype=float)

        best = None
        best_iou = 0.0
        for item in auto:
            bbox = item.get("bbox")
            if not bbox:
                continue
            bx, by, bw, bh = bbox
            cand = np.array([bx, by, bx + bw, by + bh], dtype=float)
            iou = self._bbox_iou(query, cand)
            if iou > best_iou:
                best = item
                best_iou = iou

        if best is None:
            return None

        h, w = image_rgb.shape[:2]
        polygon, bbox = self.mask_to_polygon(best["segmentation"].astype(np.uint8), w, h, epsilon)
        if polygon is None:
            return None

        return {
            "id": str(uuid.uuid4()),
            "class_name": "unlabeled",
            "score": float(best.get("predicted_iou", 0.0)),
            "area_ratio": float(np.count_nonzero(best["segmentation"])) / float(w * h),
            "polygon": polygon,
            "bbox": bbox,
            "source": "box_prompt",
            "visible": True,
        }

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        x_a = max(a[0], b[0])
        y_a = max(a[1], b[1])
        x_b = min(a[2], b[2])
        y_b = min(a[3], b[3])

        inter_w = max(0.0, x_b - x_a)
        inter_h = max(0.0, y_b - y_a)
        inter = inter_w * inter_h

        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union
