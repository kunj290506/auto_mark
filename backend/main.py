"""
Auto Annotation Tool - Backend Server
FastAPI application with WebSocket support for real-time annotation progress.
"""

import os
import re
import uuid
import asyncio
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.file_service import FileService
from services.annotation_service import AnnotationService
from services.export_service import ExportService
from session_store import save_session, load_session, delete_session as store_delete_session, list_all_sessions, session_exists

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# Initialize services
file_service = FileService(UPLOAD_DIR, TEMP_DIR)
annotation_service = AnnotationService()
export_service = ExportService(OUTPUT_DIR)

# WebSocket connections
ws_connections = {}


def extract_label_from_filename(filename: str) -> str:
    """
    Extract a clean label from an image filename.
    Examples:
        'air_conditioner_1.jpg' -> 'air conditioner'
        'applesauce_001.png' -> 'applesauce'
        'PersonWalking.jpeg' -> 'person walking'
    """
    # Remove extension
    name = Path(filename).stem
    
    # Remove trailing numbers (e.g., _1, _001, 01)
    name = re.sub(r'[_\-]?\d+$', '', name)
    
    # Handle camelCase and PascalCase
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    
    # Replace underscores and dashes with spaces
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Clean up multiple spaces and strip
    name = re.sub(r'\s+', ' ', name).strip().lower()
    
    return name if name else 'object'


class AnnotationConfig(BaseModel):
    """Configuration for annotation job"""
    objects: list[str]
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    use_sam: bool = True
    export_format: str = "coco"
    min_box_size: int = 10
    nms_threshold: float = 0.5
    remove_overlaps: bool = True


class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    status: str
    image_count: int
    processed_count: int
    annotations: Optional[dict] = None


# Bug 10: Cleanup old sessions (24hr TTL)
async def cleanup_old_sessions():
    """Remove sessions older than 24 hours"""
    while True:
        await asyncio.sleep(3600)  # Check every hour
        now = time.time()
        all_sessions = list_all_sessions()
        for sid, session in all_sessions:
            if now - session.get("created_at", now) > 86400:
                store_delete_session(sid)
                await file_service.cleanup_session(sid)
                print(f"[INFO] Cleaned up expired session: {sid}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("[INFO] Auto Annotation Tool Backend Starting...")
    print("[INFO] Upload directory:", UPLOAD_DIR.absolute())
    print("[INFO] Output directory:", OUTPUT_DIR.absolute())
    # Bug 10: Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_old_sessions())
    yield
    cleanup_task.cancel()
    print("[INFO] Shutting down...")


app = FastAPI(
    title="Auto Annotation Tool",
    description="Automated image annotation using Grounding DINO",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving processed images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Auto Annotation Tool API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "gpu_available": annotation_service.gpu_available}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a zip file containing images for annotation.
    Returns session ID and image preview information.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only zip files are supported")
    
    # Create new session
    session_id = str(uuid.uuid4())
    
    try:
        # Save and extract zip file
        result = await file_service.process_upload(file, session_id)
        
        # Bug 8: Store session with shelve + Bug 10: Add created_at
        session_data = {
            "status": "uploaded",
            "image_count": result["image_count"],
            "images": result["images"],
            "processed_count": 0,
            "annotations": {},
            "created_at": time.time()
        }
        save_session(session_id, session_data)
        
        return {
            "session_id": session_id,
            "image_count": result["image_count"],
            "images": result["images"][:20],  # Preview first 20 images
            "message": f"Successfully uploaded {result['image_count']} images"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/annotate/{session_id}")
async def start_annotation(session_id: str, config: AnnotationConfig):
    """
    Start annotation process for uploaded images.
    Uses WebSocket for real-time progress updates.
    """
    # Bug 8: Use persistent session store
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session["status"] == "processing":
        raise HTTPException(status_code=400, detail="Annotation already in progress")
    
    session["status"] = "processing"
    session["config"] = config.model_dump()
    save_session(session_id, session)
    
    # Bug 9: Start annotation in background with error callback
    task = asyncio.create_task(
        run_annotation(session_id, session["images"], config)
    )
    task.add_done_callback(
        lambda t: print(f"[ERROR] Annotation task failed: {t.exception()}") if not t.cancelled() and t.exception() else None
    )
    
    return {
        "session_id": session_id,
        "status": "processing",
        "message": "Annotation started"
    }


async def run_annotation(session_id: str, images: list, config: AnnotationConfig):
    """Background task for running annotation"""
    session = load_session(session_id)
    if session is None:
        return
    
    total = len(images)
    
    try:
        for i, image_path in enumerate(images):
            # Extract label from filename for contextual detection
            filename = os.path.basename(image_path)
            filename_label = extract_label_from_filename(filename)
            
            # Always use filename as detection label (image name IS the object)
            if config.objects and config.objects[0].lower() not in ('object', 'objects', 'item', 'items'):
                detection_labels = list(set([filename_label] + config.objects))
            else:
                detection_labels = [filename_label]
            
            # Run annotation on image with contextual label
            result = await annotation_service.annotate_image(
                image_path,
                detection_labels,
                config.box_threshold,
                config.text_threshold,
                config.use_sam,
                config.nms_threshold,
                config.min_box_size
            )
            
            # Bug 8: Store result with persistent sessions
            session = load_session(session_id)
            if session is None:
                return
            session["annotations"][image_path] = result
            session["processed_count"] = i + 1
            save_session(session_id, session)
            
            # Send WebSocket update
            if session_id in ws_connections:
                try:
                    await ws_connections[session_id].send_json({
                        "type": "progress",
                        "current": i + 1,
                        "total": total,
                        "percentage": round((i + 1) / total * 100, 1),
                        "current_image": image_path,
                        "detections": len(result.get("boxes", []))
                    })
                except Exception:
                    pass  # WebSocket may have closed
            
            # Yield to other tasks without unnecessary delay
            await asyncio.sleep(0)
        
        session = load_session(session_id)
        if session is None:
            return
        session["status"] = "completed"
        save_session(session_id, session)
        
        # Send completion message
        if session_id in ws_connections:
            try:
                await ws_connections[session_id].send_json({
                    "type": "completed",
                    "total_images": total,
                    "total_detections": sum(
                        len(ann.get("boxes", [])) 
                        for ann in session["annotations"].values()
                    )
                })
            except Exception:
                pass
            
    except Exception as e:
        session = load_session(session_id)
        if session is not None:
            session["status"] = "error"
            session["error"] = str(e)
            save_session(session_id, session)
        
        if session_id in ws_connections:
            try:
                await ws_connections[session_id].send_json({
                    "type": "error",
                    "message": str(e)
                })
            except Exception:
                pass


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    """Get current status of annotation session"""
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Calculate detections found so far
    total_detections = sum(len(ann.get("boxes", [])) for ann in session.get("annotations", {}).values())
    
    return {
        "session_id": session_id,
        "status": session["status"],
        "image_count": session["image_count"],
        "processed_count": session["processed_count"],
        "total_detections": total_detections,
        "progress": round(session["processed_count"] / session["image_count"] * 100, 1)
            if session["image_count"] > 0 else 0
    }


@app.get("/api/annotations/{session_id}")
async def get_annotations(session_id: str):
    """Get all annotations for a session"""
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "status": session["status"],
        "annotations": session["annotations"],
        "images": session["images"]
    }


@app.post("/api/export/{session_id}")
async def export_annotations(session_id: str, format: str = "coco"):
    """
    Export annotations in the specified format.
    Supported formats: coco, yolo, voc, roboflow
    """
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="Annotation not completed yet")
    
    try:
        export_path = await export_service.export(
            session_id,
            session["annotations"],
            session["images"],
            format
        )
        
        return FileResponse(
            export_path,
            media_type="application/zip",
            filename=f"annotations_{session_id}_{format}.zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/formats")
async def get_export_formats():
    """Get available export formats"""
    return {
        "formats": [
            {"id": "coco", "name": "COCO JSON", "description": "Single JSON file with all annotations"},
            {"id": "yolo", "name": "YOLO", "description": "Separate .txt files per image"},
            {"id": "voc", "name": "Pascal VOC", "description": "XML files for each image"},
            {"id": "roboflow", "name": "Roboflow", "description": "Roboflow-compatible format"}
        ]
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()
    ws_connections[session_id] = websocket
    
    try:
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        if session_id in ws_connections:
            del ws_connections[session_id]


@app.delete("/api/session/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Delete a session and its associated files"""
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        await file_service.cleanup_session(session_id)
        store_delete_session(session_id)
        return {"message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
