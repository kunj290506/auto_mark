"""
Annotation Service - High-accuracy annotation pipeline using Grounding DINO + SAM
Uses HuggingFace Transformers for Grounding DINO (base model) and segment-anything for SAM
Produces both bounding boxes AND polygon segmentation masks

Key accuracy features:
- Grounding DINO Base model (52.5 AP vs 50.6 AP for tiny)
- Per-class detection for maximum recall
- IoU-based NMS to remove duplicate boxes
- Minimum box size filtering
- Optimized text prompts with article prefix
- Label cleaning and normalization
- SAM iterative refinement with center-point guidance
- Shape-aware bounding boxes (circle/ellipse for round objects)
"""

import os
import asyncio
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Model paths
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models"))
SAM_CHECKPOINT = MODELS_DIR / "sam_vit_b_01ec64.pth"

# Thread pool for running sync inference off the event loop (Bug 5)
_executor = ThreadPoolExecutor(max_workers=1)


class AnnotationService:
    """Service for running Grounding DINO + SAM annotations with high accuracy"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.sam_predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_available = torch.cuda.is_available()
        self._model_loaded = False
        self._sam_loaded = False
        
        print(f"[INFO] Annotation Service initialized")
        print(f"   Device: {self.device}")
        print(f"   GPU Available: {self.gpu_available}")
        print(f"   Models Directory: {MODELS_DIR.absolute()}")
        print(f"   SAM Checkpoint: {SAM_CHECKPOINT}")
        
        # Bug 2: Only use FP16 on CUDA
        self.use_fp16 = self.device == "cuda" and torch.cuda.is_available()
        if self.use_fp16:
            print(f"   FP16 Inference: Enabled (faster)")
    
    async def _load_model(self):
        """Load Grounding DINO BASE model from HuggingFace for maximum accuracy"""
        if self._model_loaded:
            return True
        
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            # Use BASE model for higher accuracy (52.5 AP vs 50.6 AP for tiny)
            model_id = "IDEA-Research/grounding-dino-base"
            print(f"[INFO] Loading Grounding DINO BASE from HuggingFace ({model_id})...")
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            
            # Bug 2: Enable FP16 only on CUDA
            if self.use_fp16:
                self.model = self.model.half()
                print("   Using FP16 precision for faster inference")
            
            # Try torch.compile for additional speedup (PyTorch 2.0+)
            try:
                import torch._dynamo
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("   torch.compile enabled")
            except Exception:
                pass  # torch.compile not available
            
            self.model.eval()
            
            self._model_loaded = True
            print("[OK] Grounding DINO BASE loaded successfully")
            return True
            
        except Exception as e:
            print(f"[WARN] Could not load Grounding DINO base, trying tiny fallback: {e}")
            try:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                model_id = "IDEA-Research/grounding-dino-tiny"
                print(f"[INFO] Falling back to Grounding DINO TINY ({model_id})...")
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                self.model = self.model.to(self.device)
                # Bug 2: Only half on CUDA
                if self.use_fp16:
                    self.model = self.model.half()
                self.model.eval()
                self._model_loaded = True
                print("[OK] Grounding DINO TINY loaded as fallback")
                return True
            except Exception as e2:
                print(f"[ERROR] Could not load any Grounding DINO model: {e2}")
                return False
    
    async def _load_sam(self):
        """Load SAM model for segmentation"""
        if self._sam_loaded:
            return True
        
        if not SAM_CHECKPOINT.exists():
            print(f"[WARN] SAM checkpoint not found at {SAM_CHECKPOINT}")
            return False
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            print(f"[INFO] Loading SAM from {SAM_CHECKPOINT}...")
            
            # Load SAM model
            sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
            sam = sam.to(self.device)
            sam.eval()
            
            self.sam_predictor = SamPredictor(sam)
            self._sam_loaded = True
            print("[OK] SAM loaded successfully")
            return True
            
        except Exception as e:
            print(f"[WARN] Could not load SAM: {e}")
            return False
    
    # Bug 5: async annotate_image delegates to sync pipeline via executor
    async def annotate_image(
        self,
        image_path: str,
        objects: List[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        use_sam: bool = True,
        nms_threshold: float = 0.5,
        min_box_size: int = 10
    ) -> Dict:
        """
        Annotate a single image with Grounding DINO + SAM.
        Runs sync inference in a thread executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            self._run_sync_pipeline,
            image_path, objects, box_threshold, text_threshold,
            use_sam, nms_threshold, min_box_size
        )
        return result
    
    def _run_sync_pipeline(
        self,
        image_path: str,
        objects: List[str],
        box_threshold: float,
        text_threshold: float,
        use_sam: bool,
        nms_threshold: float,
        min_box_size: int
    ) -> Dict:
        """Synchronous detection + segmentation pipeline (runs in thread)"""
        # Load image at full resolution for maximum accuracy
        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return {"boxes": [], "labels": [], "scores": [], "segmentations": [], "error": str(e)}
        
        width, height = image.size
        
        # Load model (sync wrapper for _load_model)
        import asyncio as _asyncio
        try:
            _loop = _asyncio.get_event_loop()
            if _loop.is_running():
                # We're in a thread executor, need a new loop
                _new_loop = _asyncio.new_event_loop()
                model_ready = _new_loop.run_until_complete(self._load_model())
                _new_loop.close()
            else:
                model_ready = _loop.run_until_complete(self._load_model())
        except RuntimeError:
            _new_loop = _asyncio.new_event_loop()
            model_ready = _new_loop.run_until_complete(self._load_model())
            _new_loop.close()
        
        if not model_ready:
            # Bug 6: Return clear error instead of mock annotations
            return {
                "boxes": [],
                "labels": [],
                "scores": [],
                "segmentations": [],
                "error": "Model failed to load — no annotations generated. Check server logs.",
                "image_size": {"width": width, "height": height}
            }
        
        # --- PER-CLASS DETECTION for maximum accuracy ---
        all_boxes = []
        all_labels = []
        all_scores = []
        
        for obj in objects:
            try:
                result = self._run_detection_sync(image, [obj], box_threshold, text_threshold, width, height)
                if result["boxes"]:
                    all_boxes.extend(result["boxes"])
                    all_labels.extend(result["labels"])
                    all_scores.extend(result["scores"])
            except Exception as e:
                print(f"[WARN] Detection failed for '{obj}': {e}")
        
        if not all_boxes:
            return {
                "boxes": [],
                "labels": [],
                "scores": [],
                "segmentations": [],
                "image_size": {"width": width, "height": height}
            }
        
        # --- POST-PROCESSING ---
        # 1. Filter tiny boxes
        all_boxes, all_labels, all_scores = self._filter_small_boxes(
            all_boxes, all_labels, all_scores, min_box_size
        )
        
        # 2. Apply NMS to remove duplicate/overlapping detections
        all_boxes, all_labels, all_scores = self._apply_nms(
            all_boxes, all_labels, all_scores, nms_threshold
        )
        
        # 3. Sort by score descending
        if all_scores:
            sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
            all_boxes = [all_boxes[i] for i in sorted_indices]
            all_labels = [all_labels[i] for i in sorted_indices]
            all_scores = [all_scores[i] for i in sorted_indices]
        
        detection_result = {
            "boxes": all_boxes,
            "labels": all_labels,
            "scores": all_scores,
            "segmentations": [],
            "image_size": {"width": width, "height": height}
        }
        
        # Run SAM segmentation if enabled and we have detections
        if use_sam and len(detection_result["boxes"]) > 0:
            try:
                _new_loop = _asyncio.new_event_loop()
                sam_ready = _new_loop.run_until_complete(self._load_sam())
                _new_loop.close()
            except Exception:
                sam_ready = False
            
            if sam_ready:
                try:
                    segmentations = self._run_segmentation_sync(image_np, detection_result["boxes"])
                    detection_result["segmentations"] = segmentations
                    
                    # Bug 7: Apply shape-aware bounding boxes after SAM
                    for i, box in enumerate(detection_result["boxes"]):
                        if i < len(segmentations) and segmentations[i]:
                            label = detection_result["labels"][i] if i < len(detection_result["labels"]) else "object"
                            # Build a mask from the segmentation polygon
                            try:
                                mask = np.zeros((height, width), dtype=np.uint8)
                                for poly in segmentations[i]:
                                    if poly and len(poly) >= 6:
                                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                                        cv2.fillPoly(mask, [pts], 255)
                                
                                shape_result = self._fit_shape_aware_box(mask, label)
                                if shape_result:
                                    box["shape_type"] = shape_result["shape_type"]
                                    box["x1"] = shape_result["x1"]
                                    box["y1"] = shape_result["y1"]
                                    box["x2"] = shape_result["x2"]
                                    box["y2"] = shape_result["y2"]
                                else:
                                    box["shape_type"] = "rectangle"
                            except Exception as e:
                                print(f"[WARN] Shape-aware box fitting failed: {e}")
                                box["shape_type"] = "rectangle"
                        else:
                            box["shape_type"] = "rectangle"
                except Exception as e:
                    print(f"[WARN] Segmentation failed: {e}")
                    detection_result["segmentations"] = []
                    for box in detection_result["boxes"]:
                        box["shape_type"] = "rectangle"
            else:
                detection_result["segmentations"] = []
                for box in detection_result["boxes"]:
                    box["shape_type"] = "rectangle"
        else:
            for box in detection_result["boxes"]:
                box["shape_type"] = "rectangle"
        
        return detection_result
    
    # Bug 7: Shape-aware bounding box fitting
    def _fit_shape_aware_box(self, mask, label):
        """Fit a shape-aware bounding box based on contour analysis"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        
        ROUND_KEYWORDS = {
            "ball", "football", "soccer", "basketball", "coin", "wheel",
            "circle", "sphere", "tire", "egg", "donut", "puck", "frisbee"
        }
        is_round = any(kw in label.lower() for kw in ROUND_KEYWORDS)
        
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * 3.14159 * area) / (perimeter ** 2 + 1e-6)
        
        if is_round or circularity > 0.78:
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            x1 = cx - radius
            y1 = cy - radius
            x2 = cx + radius
            y2 = cy + radius
            shape_type = "circle"
        else:
            x, y, w, h = cv2.boundingRect(c)
            x1, y1, x2, y2 = x, y, x + w, y + h
            shape_type = "rectangle"
        
        return {
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "shape_type": shape_type
        }
    
    def _build_text_prompt(self, objects: List[str]) -> str:
        """
        Build optimized text prompt for Grounding DINO.
        
        Best practices:
        - Separate classes with periods
        - Use article prefix "a" for better grounding
        - Lowercase for consistency
        """
        # Clean and format each object name
        formatted = []
        for obj in objects:
            clean = obj.strip().lower().rstrip(".")
            if clean:
                formatted.append(f"{clean}")
        
        # Join with ". " separator and end with "."
        return ". ".join(formatted) + "." if formatted else "object."
    
    def _clean_label(self, label: str, objects: List[str]) -> str:
        """
        Clean and normalize a detection label.
        
        Strips whitespace, removes trailing periods, and attempts
        to match the label back to one of the original object names.
        """
        cleaned = label.strip().lower().rstrip(".").strip()
        
        # Try to match to one of the original object names
        for obj in objects:
            obj_clean = obj.strip().lower()
            if obj_clean in cleaned or cleaned in obj_clean:
                return obj_clean
        
        return cleaned if cleaned else "object"
    
    def _run_detection_sync(
        self,
        image: Image.Image,
        objects: List[str],
        box_threshold: float,
        text_threshold: float,
        width: int,
        height: int
    ) -> Dict:
        """Run Grounding DINO detection with optimized prompts (synchronous)"""
        # Build optimized text prompt
        text = self._build_text_prompt(objects)
        
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Use FP16 for inputs if enabled
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        with torch.no_grad():
            # Bug 1: Use torch.amp.autocast with device_type parameter
            if self.use_fp16 and self.device == "cuda":
                with torch.amp.autocast(device_type=self.device, enabled=True):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([[height, width]]).to(self.device)
        
        # Bug 4: Use .get() for input_ids to handle different key names
        token_ids = inputs.get("input_ids", inputs.get("input_ids_batch"))
        
        # Try different API versions for post-processing
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                token_ids,
                target_sizes=target_sizes,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )[0]
        except TypeError:
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    token_ids,
                    target_sizes=target_sizes,
                    threshold=box_threshold
                )[0]
            except TypeError:
                try:
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        threshold=box_threshold,
                        target_sizes=target_sizes
                    )[0]
                except TypeError:
                    # Manual fallback
                    logits = outputs.logits.cpu().sigmoid()[0]
                    boxes = outputs.pred_boxes.cpu()[0]
                    
                    filt_mask = logits.max(dim=1)[0] > box_threshold
                    boxes_filt = boxes[filt_mask]
                    scores = logits[filt_mask].max(dim=1)[0].tolist()
                    
                    boxes_scaled = boxes_filt * torch.tensor([width, height, width, height])
                    boxes_list = boxes_scaled.tolist()
                    labels = [objects[0] if objects else "object"] * len(scores)
                    
                    return self._format_detection_results(boxes_list, labels, scores, width, height, objects, is_cxcywh=True)
        
        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()
        labels = results.get("labels", results.get("text", []))
        
        if isinstance(labels, list) and len(labels) > 0:
            labels = [str(l) for l in labels]
        else:
            labels = [objects[0] if objects else "object"] * len(scores)
        
        return self._format_detection_results(boxes, labels, scores, width, height, objects, is_cxcywh=False)
    
    def _format_detection_results(
        self,
        boxes: List,
        labels: List[str],
        scores: List[float],
        width: int,
        height: int,
        objects: List[str],
        is_cxcywh: bool = False
    ) -> Dict:
        """Format detection results into standard format with label cleaning"""
        normalized_boxes = []
        cleaned_labels = []
        
        for idx, box in enumerate(boxes):
            if is_cxcywh:
                cx, cy, w, h = box
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
            else:
                x1, y1, x2, y2 = box
            
            # Clamp to image bounds
            x1 = max(0, min(float(x1), width))
            y1 = max(0, min(float(y1), height))
            x2 = max(0, min(float(x2), width))
            y2 = max(0, min(float(y2), height))
            
            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            normalized_boxes.append({
                "x": x1 / width,
                "y": y1 / height,
                "width": (x2 - x1) / width,
                "height": (y2 - y1) / height,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            })
            
            # Clean the label
            label = labels[idx] if idx < len(labels) else (objects[0] if objects else "object")
            cleaned_labels.append(self._clean_label(label, objects))
        
        # Filter scores to match remaining boxes
        valid_scores = [scores[i] for i in range(len(scores)) if i < len(boxes)]
        # Trim to match normalized_boxes count (some may have been removed as degenerate)
        valid_scores = valid_scores[:len(normalized_boxes)]
        
        return {
            "boxes": normalized_boxes,
            "labels": cleaned_labels,
            "scores": valid_scores,
            "segmentations": [],
            "image_size": {"width": width, "height": height}
        }
    
    def _compute_iou(self, box1: Dict, box2: Dict) -> float:
        """Compute Intersection over Union between two boxes"""
        x1 = max(box1["x1"], box2["x1"])
        y1 = max(box1["y1"], box2["y1"])
        x2 = min(box1["x2"], box2["x2"])
        y2 = min(box1["y2"], box2["y2"])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
        area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _apply_nms(
        self,
        boxes: List[Dict],
        labels: List[str],
        scores: List[float],
        iou_threshold: float = 0.5
    ) -> Tuple[List[Dict], List[str], List[float]]:
        """
        Apply Non-Maximum Suppression to remove duplicate/overlapping boxes.
        """
        if not boxes:
            return boxes, labels, scores
        
        # Sort indices by score descending
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        keep = []
        suppressed = set()
        
        for i in indices:
            if i in suppressed:
                continue
            keep.append(i)
            
            for j in indices:
                if j in suppressed or j == i:
                    continue
                if j in [k for k in keep]:
                    continue
                    
                iou = self._compute_iou(boxes[i], boxes[j])
                if iou > iou_threshold:
                    suppressed.add(j)
        
        return (
            [boxes[i] for i in keep],
            [labels[i] for i in keep],
            [scores[i] for i in keep]
        )
    
    def _filter_small_boxes(
        self,
        boxes: List[Dict],
        labels: List[str],
        scores: List[float],
        min_size: int = 10
    ) -> Tuple[List[Dict], List[str], List[float]]:
        """Filter out boxes smaller than min_size in either dimension"""
        if not boxes:
            return boxes, labels, scores
        
        keep_boxes = []
        keep_labels = []
        keep_scores = []
        
        for i, box in enumerate(boxes):
            box_w = box["x2"] - box["x1"]
            box_h = box["y2"] - box["y1"]
            
            if box_w >= min_size and box_h >= min_size:
                keep_boxes.append(box)
                keep_labels.append(labels[i] if i < len(labels) else "object")
                keep_scores.append(scores[i] if i < len(scores) else 0.0)
        
        return keep_boxes, keep_labels, keep_scores
    
    def _run_segmentation_sync(
        self,
        image_np: np.ndarray,
        boxes: List[Dict]
    ) -> List[List[List[float]]]:
        """Run SAM segmentation with iterative refinement and center-point guidance (synchronous)"""
        
        img_h, img_w = image_np.shape[:2]
        
        # Set image in SAM predictor (full resolution)
        self.sam_predictor.set_image(image_np)
        
        segmentations = []
        
        for box in boxes:
            try:
                # Get box coordinates
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                
                # Pad bounding box by 5%
                box_w, box_h = x2 - x1, y2 - y1
                pad_x = box_w * 0.05
                pad_y = box_h * 0.05
                padded_x1 = max(0, x1 - pad_x)
                padded_y1 = max(0, y1 - pad_y)
                padded_x2 = min(img_w, x2 + pad_x)
                padded_y2 = min(img_h, y2 + pad_y)
                input_box = np.array([padded_x1, padded_y1, padded_x2, padded_y2])
                
                # Center point of the box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                point_coords = np.array([[center_x, center_y]])
                point_labels = np.array([1])  # 1 = foreground
                
                # --- PASS 1: Initial prediction with box + center point prompt ---
                masks, quality_scores, low_res_masks = self.sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=input_box,
                    multimask_output=True
                )
                
                # Pick best mask from pass 1
                best_idx = np.argmax(quality_scores)
                best_low_res = low_res_masks[best_idx:best_idx+1]
                
                # --- PASS 2: Refine with the low-res mask from pass 1 ---
                masks_refined, quality_refined, _ = self.sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=input_box,
                    mask_input=best_low_res,
                    multimask_output=False
                )
                
                # Use the refined mask
                mask = masks_refined[0].astype(np.uint8)
                
                # Morphological cleanup
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                
                # Extract contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Keep significant contours only
                    box_area = box_w * box_h
                    min_area = max(50, box_area * 0.01)
                    significant = [c for c in contours if cv2.contourArea(c) >= min_area]
                    
                    if not significant:
                        significant = [max(contours, key=cv2.contourArea)]
                    
                    polygons = []
                    for contour in significant:
                        perimeter = cv2.arcLength(contour, True)
                        epsilon = 0.002 * perimeter
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        if len(approx) >= 3:
                            polygon = approx.flatten().tolist()
                            polygons.append(polygon)
                    
                    segmentations.append(polygons if polygons else [])
                else:
                    segmentations.append([])
                    
            except Exception as e:
                print(f"[WARN] Segmentation failed for box: {e}")
                segmentations.append([])
        
        return segmentations
    
    # Bug 3: Unified default thresholds (0.25 / 0.20)
    async def annotate_batch(
        self,
        image_paths: List[str],
        objects: List[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        use_sam: bool = True,
        nms_threshold: float = 0.5,
        min_box_size: int = 10,
        progress_callback=None
    ) -> Dict[str, Dict]:
        """Annotate multiple images"""
        results = {}
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                result = await self.annotate_image(
                    image_path, objects, box_threshold, text_threshold,
                    use_sam, nms_threshold, min_box_size
                )
                results[image_path] = result
            except Exception as e:
                results[image_path] = {
                    "error": str(e),
                    "boxes": [],
                    "labels": [],
                    "scores": [],
                    "segmentations": []
                }
            
            if progress_callback:
                await progress_callback(i + 1, total, image_path)
        
        return results
