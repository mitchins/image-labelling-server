"""
Media helpers shared by ingest and serving paths.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
}

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
}

MEDIA_EXTENSIONS = {
    "image": IMAGE_EXTENSIONS,
    "audio": AUDIO_EXTENSIONS,
}


def normalize_media_type(media_type: str | None) -> str:
    value = (media_type or "image").strip().lower()
    if value not in MEDIA_EXTENSIONS:
        raise ValueError(f"Unsupported media_type: {media_type}")
    return value


def collect_media(root: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    if recursive:
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in extensions]


def guess_media_type_from_path(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    return "image"


def guess_mime_type(path: str | Path, media_type: str | None = None) -> str:
    normalized = normalize_media_type(media_type) if media_type else guess_media_type_from_path(path)
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    return "audio/mpeg" if normalized == "audio" else "image/jpeg"
