"""
Export Service - Convert annotations to multiple formats (COCO, YOLO, VOC, Roboflow)
"""

import os
import json
import zipfile
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom


class ExportService:
    """Service for exporting annotations in various formats"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    async def export(
        self,
        session_id: str,
        annotations: Dict[str, Dict],
        images: List[str],
        format: str = "coco"
    ) -> str:
        """
        Export annotations in the specified format.
        
        Args:
            session_id: Session identifier
            annotations: Dictionary of image paths to annotation data
            images: List of image paths
            format: Export format (coco, yolo, voc, roboflow)
        
        Returns:
            Path to the exported zip file
        """
        export_dir = self.output_dir / session_id / format
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Get class names
        class_names = self._extract_class_names(annotations)
        
        if format == "coco":
            await self._export_coco(export_dir, annotations, images, class_names)
        elif format == "yolo":
            await self._export_yolo(export_dir, annotations, images, class_names)
        elif format == "voc":
            await self._export_voc(export_dir, annotations, images, class_names)
        elif format == "roboflow":
            await self._export_roboflow(export_dir, annotations, images, class_names)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Create metadata
        await self._create_metadata(export_dir, session_id, format, class_names, len(images))
        
        # Create zip file
        zip_path = self.output_dir / f"{session_id}_{format}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(export_dir)
                    zipf.write(file_path, arcname)
        
        return str(zip_path)
    
    def _extract_class_names(self, annotations: Dict) -> List[str]:
        """Extract unique class names from annotations"""
        classes = set()
        for ann in annotations.values():
            if "labels" in ann:
                classes.update(ann["labels"])
        return sorted(list(classes))
    
    async def _export_coco(
        self,
        export_dir: Path,
        annotations: Dict,
        images: List[str],
        class_names: List[str]
    ):
        """Export in COCO JSON format with segmentation support"""
        
        coco_data = {
            "info": {
                "description": "Auto Annotation Tool Export",
                "date_created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "licenses": [],
            "categories": [
                {"id": i + 1, "name": name, "supercategory": "object"}
                for i, name in enumerate(class_names)
            ],
            "images": [],
            "annotations": []
        }
        
        annotation_id = 1
        
        for image_id, image_path in enumerate(images, 1):
            path = Path(image_path)
            ann = annotations.get(image_path, {})
            
            # Get image size
            size = ann.get("image_size", {"width": 1920, "height": 1080})
            
            coco_data["images"].append({
                "id": image_id,
                "file_name": path.name,
                "width": size["width"],
                "height": size["height"]
            })
            
            # Add annotations
            boxes = ann.get("boxes", [])
            labels = ann.get("labels", [])
            scores = ann.get("scores", [])
            segmentations = ann.get("segmentations", [])
            
            for i, box in enumerate(boxes):
                label = labels[i] if i < len(labels) else "unknown"
                score = scores[i] if i < len(scores) else 1.0
                
                category_id = class_names.index(label) + 1 if label in class_names else 1
                
                # Convert normalized to pixel coordinates
                x = box["x"] * size["width"]
                y = box["y"] * size["height"]
                w = box["width"] * size["width"]
                h = box["height"] * size["height"]
                
                # Build annotation entry
                ann_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": score
                }
                
                # Add segmentation if available
                if i < len(segmentations) and segmentations[i]:
                    ann_entry["segmentation"] = segmentations[i]
                    # Recalculate area from segmentation polygon
                    if segmentations[i] and len(segmentations[i]) > 0:
                        poly = segmentations[i][0]
                        if len(poly) >= 6:  # At least 3 points
                            # Calculate polygon area using shoelace formula
                            n = len(poly) // 2
                            area = 0
                            for j in range(n):
                                x1, y1 = poly[2*j], poly[2*j + 1]
                                x2, y2 = poly[2*((j+1) % n)], poly[2*((j+1) % n) + 1]
                                area += x1 * y2 - x2 * y1
                            ann_entry["area"] = abs(area) / 2
                else:
                    # Use bbox as segmentation fallback (4 corner points)
                    x1, y1 = box.get("x1", x), box.get("y1", y)
                    x2, y2 = box.get("x2", x + w), box.get("y2", y + h)
                    ann_entry["segmentation"] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                
                coco_data["annotations"].append(ann_entry)
                annotation_id += 1
        
        # Save COCO JSON
        with open(export_dir / "annotations.json", 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    async def _export_yolo(
        self,
        export_dir: Path,
        annotations: Dict,
        images: List[str],
        class_names: List[str]
    ):
        """Export in YOLO format (separate .txt files)"""
        
        labels_dir = export_dir / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        # Create classes.txt
        with open(export_dir / "classes.txt", 'w') as f:
            f.write('\n'.join(class_names))
        
        # Create data.yaml
        data_yaml = {
            "path": ".",
            "train": "images",
            "val": "images",
            "names": {i: name for i, name in enumerate(class_names)}
        }
        with open(export_dir / "data.yaml", 'w') as f:
            import yaml
            yaml.dump(data_yaml, f)
        
        # Create label files
        for image_path in images:
            path = Path(image_path)
            ann = annotations.get(image_path, {})
            
            boxes = ann.get("boxes", [])
            labels = ann.get("labels", [])
            
            label_file = labels_dir / f"{path.stem}.txt"
            
            with open(label_file, 'w') as f:
                for i, box in enumerate(boxes):
                    label = labels[i] if i < len(labels) else "unknown"
                    class_id = class_names.index(label) if label in class_names else 0
                    
                    # YOLO format: class_id center_x center_y width height (normalized)
                    cx = box["x"] + box["width"] / 2
                    cy = box["y"] + box["height"] / 2
                    w = box["width"]
                    h = box["height"]
                    
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    async def _export_voc(
        self,
        export_dir: Path,
        annotations: Dict,
        images: List[str],
        class_names: List[str]
    ):
        """Export in Pascal VOC XML format"""
        
        annotations_dir = export_dir / "Annotations"
        annotations_dir.mkdir(exist_ok=True)
        
        for image_path in images:
            path = Path(image_path)
            ann = annotations.get(image_path, {})
            size = ann.get("image_size", {"width": 1920, "height": 1080})
            
            # Create XML structure
            annotation = ET.Element("annotation")
            
            ET.SubElement(annotation, "folder").text = "images"
            ET.SubElement(annotation, "filename").text = path.name
            
            size_elem = ET.SubElement(annotation, "size")
            ET.SubElement(size_elem, "width").text = str(size["width"])
            ET.SubElement(size_elem, "height").text = str(size["height"])
            ET.SubElement(size_elem, "depth").text = "3"
            
            ET.SubElement(annotation, "segmented").text = "0"
            
            boxes = ann.get("boxes", [])
            labels = ann.get("labels", [])
            scores = ann.get("scores", [])
            
            for i, box in enumerate(boxes):
                label = labels[i] if i < len(labels) else "unknown"
                
                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = label
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                
                if i < len(scores):
                    ET.SubElement(obj, "confidence").text = str(scores[i])
                
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(box["x1"]))
                ET.SubElement(bndbox, "ymin").text = str(int(box["y1"]))
                ET.SubElement(bndbox, "xmax").text = str(int(box["x2"]))
                ET.SubElement(bndbox, "ymax").text = str(int(box["y2"]))
            
            # Pretty print XML
            xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
            
            with open(annotations_dir / f"{path.stem}.xml", 'w') as f:
                f.write(xml_str)
    
    async def _export_roboflow(
        self,
        export_dir: Path,
        annotations: Dict,
        images: List[str],
        class_names: List[str]
    ):
        """Export in Roboflow-compatible format"""
        
        # Roboflow uses COCO format with additional metadata
        await self._export_coco(export_dir, annotations, images, class_names)
        
        # Add Roboflow-specific metadata
        roboflow_meta = {
            "version": "1.0",
            "type": "object_detection",
            "classes": class_names,
            "export_format": "coco",
            "created_by": "Auto Annotation Tool",
            "created_at": datetime.now().isoformat()
        }
        
        with open(export_dir / "_roboflow.json", 'w') as f:
            json.dump(roboflow_meta, f, indent=2)
    
    async def _create_metadata(
        self,
        export_dir: Path,
        session_id: str,
        format: str,
        class_names: List[str],
        image_count: int
    ):
        """Create metadata file"""
        
        metadata = {
            "session_id": session_id,
            "export_format": format,
            "class_names": class_names,
            "image_count": image_count,
            "exported_at": datetime.now().isoformat(),
            "tool": "Auto Annotation Tool",
            "version": "1.0.0"
        }
        
        with open(export_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create README
        readme = f"""# Exported Annotations

- **Session ID**: {session_id}
- **Format**: {format.upper()}
- **Images**: {image_count}
- **Classes**: {', '.join(class_names)}
- **Exported**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage

"""
        
        if format == "coco":
            readme += """This dataset is in COCO format. Load with:

```python
from pycocotools.coco import COCO
coco = COCO('annotations.json')
```
"""
        elif format == "yolo":
            readme += """This dataset is in YOLO format. Use with:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data.yaml')
```
"""
        
        with open(export_dir / "README.md", 'w') as f:
            f.write(readme)


# YAML support for YOLO export
try:
    import yaml
except ImportError:
    # Simple YAML writer fallback
    class yaml:
        @staticmethod
        def dump(data, f):
            for key, value in data.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")
