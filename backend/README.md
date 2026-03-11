# AutoMark SAM2 Backend

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## API Endpoints

- `POST /api/projects` - Create project
- `GET /api/projects` - List projects
- `POST /api/projects/{project_id}/upload/images` - Upload images
- `POST /api/projects/{project_id}/upload/zip` - Upload ZIP
- `POST /api/projects/{project_id}/annotate/auto` - Auto-annotate one image with SAM2
- `POST /api/projects/{project_id}/annotate/auto/batch` - Batch auto-annotation
- `GET /api/jobs/{job_id}` - Batch progress + ETA
- `POST /api/projects/{project_id}/annotate/point` - Point prompt annotation
- `POST /api/projects/{project_id}/annotate/box` - Box prompt annotation
- `PUT /api/projects/{project_id}/annotations/{image_id}` - Save edited polygons
- `POST /api/projects/{project_id}/export` - Build YOLO dataset + COCO export
- `GET /api/projects/{project_id}/export/download` - Download export ZIP
