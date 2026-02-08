# Auto Annotation Tool - Backend

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## API Endpoints

- `POST /api/upload` - Upload zip file
- `POST /api/annotate` - Start annotation
- `GET /api/status/{session_id}` - Get processing status
- `GET /api/export/{session_id}` - Download annotations
- `WS /ws/{session_id}` - Real-time progress updates
