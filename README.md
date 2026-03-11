# AutoMark SAM2 Auto-Annotation Tool

AutoMark is a Python + React auto-annotation platform that uses **SAM2 (Segment Anything Model v2)** for full-image object mask generation and prompt-based mask refinement, then exports directly to training-ready YOLO datasets.

## What This Build Includes

- FastAPI backend with project-based local storage.
- SAM2 automatic mask generation for all objects in an image.
- Point and box prompt segmentation.
- Polygon mask editing workflow with undo/redo.
- Class management (add, rename, delete, merge, shortcuts).
- Confidence filtering and per-image status tracking.
- Batch auto-annotation with progress/ETA and skip-on-error behavior.
- YOLO export (detect and segment) with required folder structure and `data.yaml`.
- COCO JSON export.
- Optional augmentation pipeline before export (flip, rotate, jitter, mosaic, resize).
- Dataset stats dashboard and imbalance warnings.

## Tech Stack

- Backend: FastAPI + Python
- Segmentation model: SAM2
- Frontend: React + Vite (dark mode single-page workspace)
- Storage: Local filesystem (`backend/projects/...`)

## 1. Backend Setup

1. Open a terminal in `backend`.
2. Install dependencies.
3. Start the API.

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The backend runs on `http://localhost:8000`.

### SAM2 Model Files

Place SAM2 files in `backend/models`:

- `sam2_hiera_large.pt`
- `sam2_hiera_large.yaml`

Default runtime settings are in `backend/services/sam2_service.py`.

## 2. Frontend Setup

```bash
cd frontend-app
npm install
npm run dev
```

Open `http://localhost:5173`.

## 3. Primary API Endpoints

- `POST /api/projects`
- `GET /api/projects`
- `POST /api/projects/{project_id}/upload/images`
- `POST /api/projects/{project_id}/upload/zip`
- `POST /api/projects/{project_id}/annotate/auto`
- `POST /api/projects/{project_id}/annotate/auto/batch`
- `POST /api/projects/{project_id}/annotate/point`
- `POST /api/projects/{project_id}/annotate/box`
- `PUT /api/projects/{project_id}/annotations/{image_id}`
- `POST /api/projects/{project_id}/export`
- `GET /api/projects/{project_id}/export/download`
- `GET /api/projects/{project_id}/export/coco`

## 4. YOLO Export Structure

The export writes:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

`data.yaml` follows YOLOv5/YOLOv8/YOLOv11 conventions.

## 5. Example: Annotate 100 Images And Train YOLOv8

1. Create a project in the UI.
2. Upload a folder of 100 images (or one ZIP with all 100 files).
3. Click **Batch Auto-Annotate**.
4. Wait for progress to reach 100%.
5. Review masks image-by-image; fix classes and polygons where needed.
6. Click **Export YOLO**.
7. Unzip export and train with Ultralytics:

```bash
yolo task=segment mode=train model=yolov8n-seg.pt data=dataset/data.yaml epochs=100 imgsz=640
```

Use `task=detect` and a detect model if you exported detection labels.

## Notes

- If SAM2 fails for one image during batch mode, that image is skipped and logged in `backend/projects/<project_id>/logs/errors.log`.
- Annotation JSON is persisted per image in `backend/projects/<project_id>/annotations`, so refresh/resume is supported.
