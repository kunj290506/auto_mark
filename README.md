# AutoMark - AI-Powered Annotation Tool

A premium auto-annotation web application using **Grounding DINO** for zero-shot object detection.

![AutoMark](https://img.shields.io/badge/Version-1.0.0-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![React](https://img.shields.io/badge/React-18+-cyan)

## Features

- 🎯 **Zero-Shot Detection** - Describe what you want to find, no training needed
- 📦 **Multiple Export Formats** - COCO, YOLO, Pascal VOC, Roboflow
- ⚡ **Fast Processing** - 2-5 seconds per image with GPU
- 🎨 **Premium UI** - Modern dark theme with glassmorphism
- 📊 **Real-time Progress** - WebSocket-powered live updates
- 🖼️ **Visual Review** - Canvas-based annotation viewer with editing

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- GPU recommended (works with CPU)

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

```bash
cd frontend-app
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## Workflow

1. **Upload** - Drag and drop a ZIP file containing images
2. **Configure** - Define objects to detect and adjust thresholds
3. **Process** - Watch AI annotate your images in real-time
4. **Review** - Visualize and edit annotations
5. **Export** - Download in your preferred format

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18 + Vite |
| Backend | FastAPI + WebSocket |
| ML Model | Grounding DINO |
| Styling | Custom CSS with glassmorphism |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload zip file |
| `/api/annotate/{session}` | POST | Start annotation |
| `/api/status/{session}` | GET | Get progress |
| `/api/export/{session}` | POST | Download annotations |
| `/ws/{session}` | WS | Real-time updates |

## Export Formats

- **COCO JSON** - Industry standard, single file
- **YOLO** - Separate .txt files, normalized coordinates
- **Pascal VOC** - XML format for each image
- **Roboflow** - Platform-ready format

## Project Structure

```
automark/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt     # Python dependencies
│   └── services/
│       ├── file_service.py      # File handling
│       ├── annotation_service.py # Grounding DINO
│       └── export_service.py     # Format exporters
│
├── frontend-app/
│   ├── src/
│   │   ├── App.jsx          # Main app
│   │   ├── components/      # UI components
│   │   └── pages/           # Page components
│   └── index.html
│
└── IMPLEMENTATION_GUIDE.md  # Detailed documentation
```

## License

MIT License
