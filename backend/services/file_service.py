"""
File Service - Handle file uploads, extraction, and storage
"""

import os
import zipfile
import shutil
import aiofiles
from pathlib import Path
from fastapi import UploadFile
from typing import List, Dict


class FileService:
    """Service for handling file operations"""
    
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
    
    def __init__(self, upload_dir: Path, temp_dir: Path):
        self.upload_dir = upload_dir
        self.temp_dir = temp_dir
    
    async def process_upload(self, file: UploadFile, session_id: str) -> Dict:
        """
        Process uploaded zip file:
        1. Save to temp directory
        2. Extract contents
        3. Validate and organize images
        4. Return image list
        """
        session_dir = self.upload_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save zip file
        zip_path = self.temp_dir / f"{session_id}.zip"
        
        async with aiofiles.open(zip_path, 'wb') as f:
            content = await file.read()
            if len(content) > self.MAX_FILE_SIZE:
                raise ValueError(f"File too large. Maximum size is 1GB")
            await f.write(content)
        
        # Extract zip file
        images = []
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    # Skip directories and hidden files
                    if member.endswith('/') or member.startswith('__MACOSX'):
                        continue
                    
                    # Check if it's an image
                    ext = Path(member).suffix.lower()
                    if ext in self.ALLOWED_EXTENSIONS:
                        # Extract file
                        filename = Path(member).name
                        target_path = session_dir / filename
                        
                        # Handle duplicate names
                        counter = 1
                        while target_path.exists():
                            stem = Path(filename).stem
                            target_path = session_dir / f"{stem}_{counter}{ext}"
                            counter += 1
                        
                        with zip_ref.open(member) as source:
                            with open(target_path, 'wb') as target:
                                target.write(source.read())
                        
                        images.append(str(target_path))
            
        finally:
            # Clean up zip file
            if zip_path.exists():
                os.remove(zip_path)
        
        if not images:
            raise ValueError("No valid images found in zip file")
        
        return {
            "image_count": len(images),
            "images": sorted(images),
            "session_dir": str(session_dir)
        }
    
    async def get_image_url(self, image_path: str, session_id: str) -> str:
        """Get relative URL for an image"""
        path = Path(image_path)
        return f"/uploads/{session_id}/{path.name}"
    
    async def cleanup_session(self, session_id: str):
        """Clean up all files associated with a session"""
        session_dir = self.upload_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        
        # Clean up any temp files
        zip_path = self.temp_dir / f"{session_id}.zip"
        if zip_path.exists():
            os.remove(zip_path)
    
    def validate_image(self, image_path: str) -> bool:
        """Validate that file is a valid image"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    async def get_image_info(self, image_path: str) -> Dict:
        """Get image metadata"""
        from PIL import Image
        with Image.open(image_path) as img:
            return {
                "path": image_path,
                "filename": Path(image_path).name,
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }
