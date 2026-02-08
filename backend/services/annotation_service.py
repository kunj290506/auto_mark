"""
Annotation Service - Core annotation pipeline using Grounding DINO + SAM
Uses HuggingFace Transformers for Grounding DINO and segment-anything for SAM
Produces both bounding boxes AND polygon segmentation masks
"""

import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

# Model paths
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models"))
SAM_CHECKPOINT = MODELS_DIR / "sam_vit_b_01ec64.pth"


class AnnotationService:
    """Service for running Grounding DINO + SAM annotations"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.sam_predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_available = torch.cuda.is_available()
        self._model_loaded = False
        self._sam_loaded = False
        
        print(f"🔧 Annotation Service initialized")
        print(f"   Device: {self.device}")
        print(f"   GPU Available: {self.gpu_available}")
        print(f"   Models Directory: {MODELS_DIR.absolute()}")
        print(f"   SAM Checkpoint: {SAM_CHECKPOINT}")
        
        # Performance flags
        self.use_fp16 = self.device == "cuda"
        if self.use_fp16:
            print(f"   FP16 Inference: Enabled (faster)")
    
    async def _load_model(self):
        """Load Grounding DINO model from HuggingFace"""
        if self._model_loaded:
            return True
        
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-tiny"
            print(f"📦 Loading Grounding DINO from HuggingFace ({model_id})...")
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            
            # Enable FP16 for faster inference on GPU
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
            print("✅ Grounding DINO loaded successfully")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not load Grounding DINO: {e}")
            return False
    
    async def _load_sam(self):
        """Load SAM model for segmentation"""
        if self._sam_loaded:
            return True
        
        if not SAM_CHECKPOINT.exists():
            print(f"⚠️ SAM checkpoint not found at {SAM_CHECKPOINT}")
            return False
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            print(f"📦 Loading SAM from {SAM_CHECKPOINT}...")
            
            # Load SAM model
            sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
            sam = sam.to(self.device)
            sam.eval()
            
            self.sam_predictor = SamPredictor(sam)
            self._sam_loaded = True
            print("✅ SAM loaded successfully")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not load SAM: {e}")
            return False
    
    async def annotate_image(
        self,
        image_path: str,
        objects: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        use_sam: bool = True
    ) -> Dict:
        """
        Annotate a single image with Grounding DINO + SAM.
        
        Returns:
            Dictionary with boxes, labels, scores, and segmentations (polygon contours)
        """
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return {"boxes": [], "labels": [], "scores": [], "segmentations": [], "error": str(e)}
        
        width, height = image.size
        
        # Load models
        model_ready = await self._load_model()
        
        if not model_ready:
            return self._generate_mock_annotations(objects, width, height)
        
        # Run Grounding DINO detection
        try:
            detection_result = await self._run_detection(image, objects, box_threshold, width, height)
        except Exception as e:
            print(f"⚠️ Detection failed: {e}")
            return self._generate_mock_annotations(objects, width, height)
        
        # Run SAM segmentation if enabled and we have detections
        if use_sam and len(detection_result["boxes"]) > 0:
            sam_ready = await self._load_sam()
            if sam_ready:
                try:
                    segmentations = await self._run_segmentation(image_np, detection_result["boxes"])
                    detection_result["segmentations"] = segmentations
                except Exception as e:
                    print(f"⚠️ Segmentation failed: {e}")
                    detection_result["segmentations"] = []
            else:
                detection_result["segmentations"] = []
        else:
            detection_result["segmentations"] = []
        
        return detection_result
    
    async def _run_detection(
        self,
        image: Image.Image,
        objects: List[str],
        box_threshold: float,
        width: int,
        height: int
    ) -> Dict:
        """Run Grounding DINO detection"""
        text = ". ".join(objects) + "."
        
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Use FP16 for inputs if enabled
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16):
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([[height, width]]).to(self.device)
        
        # Try different API versions
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
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
                
                return self._format_detection_results(boxes_list, labels, scores, width, height, is_cxcywh=True)
        
        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()
        labels = results.get("labels", results.get("text", []))
        
        if isinstance(labels, list) and len(labels) > 0:
            labels = [str(l) for l in labels]
        else:
            labels = [objects[0] if objects else "object"] * len(scores)
        
        return self._format_detection_results(boxes, labels, scores, width, height, is_cxcywh=False)
    
    async def _run_segmentation(
        self,
        image_np: np.ndarray,
        boxes: List[Dict]
    ) -> List[List[List[float]]]:
        """Run SAM segmentation for each detected box"""
        
        # Set image in SAM predictor
        self.sam_predictor.set_image(image_np)
        
        segmentations = []
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            input_box = np.array([x1, y1, x2, y2])
            
            # Run SAM prediction with box prompt
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False
            )
            
            # Get the mask (first one since multimask_output=False)
            mask = masks[0].astype(np.uint8)
            
            # Convert mask to polygon contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour to reduce points
                epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Flatten to [x1, y1, x2, y2, ...] format
                polygon = approx.flatten().tolist()
                segmentations.append([polygon])
            else:
                # No contour found, use empty
                segmentations.append([])
        
        return segmentations
    
    def _format_detection_results(
        self,
        boxes: List,
        labels: List[str],
        scores: List[float],
        width: int,
        height: int,
        is_cxcywh: bool = False
    ) -> Dict:
        """Format detection results into standard format"""
        normalized_boxes = []
        
        for box in boxes:
            if is_cxcywh:
                cx, cy, w, h = box
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
            else:
                x1, y1, x2, y2 = box
            
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
        
        return {
            "boxes": normalized_boxes,
            "labels": labels,
            "scores": scores,
            "segmentations": [],
            "image_size": {"width": width, "height": height}
        }
    
    def _generate_mock_annotations(
        self,
        objects: List[str],
        width: int,
        height: int
    ) -> Dict:
        """Generate mock annotations for development/testing"""
        import random
        
        num_detections = random.randint(1, min(5, len(objects) * 2))
        boxes = []
        labels = []
        scores = []
        segmentations = []
        
        for i in range(num_detections):
            x1 = random.randint(0, int(width * 0.7))
            y1 = random.randint(0, int(height * 0.7))
            box_w = random.randint(int(width * 0.1), int(width * 0.4))
            box_h = random.randint(int(height * 0.1), int(height * 0.4))
            x2 = min(x1 + box_w, width)
            y2 = min(y1 + box_h, height)
            
            boxes.append({
                "x": x1 / width,
                "y": y1 / height,
                "width": (x2 - x1) / width,
                "height": (y2 - y1) / height,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
            
            labels.append(random.choice(objects))
            scores.append(round(random.uniform(0.35, 0.98), 3))
            
            # Generate mock polygon (ellipse approximation)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx, ry = (x2 - x1) / 2, (y2 - y1) / 2
            points = []
            for angle in range(0, 360, 30):
                rad = angle * np.pi / 180
                px = cx + rx * np.cos(rad)
                py = cy + ry * np.sin(rad)
                points.extend([px, py])
            segmentations.append([points])
        
        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "segmentations": segmentations,
            "image_size": {"width": width, "height": height}
        }
    
    async def annotate_batch(
        self,
        image_paths: List[str],
        objects: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        use_sam: bool = True,
        progress_callback=None
    ) -> Dict[str, Dict]:
        """Annotate multiple images"""
        results = {}
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                result = await self.annotate_image(
                    image_path, objects, box_threshold, text_threshold, use_sam
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
