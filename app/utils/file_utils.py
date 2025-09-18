import aiofiles
import mimetypes
from pathlib import Path
from typing import Dict, Any


async def save_uploaded_file(content: bytes, file_path: Path):
    """Save the uploaded file to disk"""
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get file information"""
    stat = file_path.stat()
    mime_type, _ = mimetypes.guess_type(str(file_path))

    return {
        "size": stat.st_size,
        "mime_type": mime_type or "application/octet-stream",
        "extension": file_path.suffix.lower()
    }


def is_allowed_file(filename: str, allowed_extensions: list) -> bool:
    """Check whether the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def get_file_size_mb(size_bytes: int) -> float:
    """Convert size to MB"""
    return round(size_bytes / (1024 * 1024), 2)