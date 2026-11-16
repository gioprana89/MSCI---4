"""Reliable, low-memory download utilities for large Streamlit research outputs.

The file is never modified; fallback parts are byte-identical slices of the ZIP/XLSX.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import math

CHUNK_BYTES = 12 * 1024 * 1024


def sha256_file(path: str | Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def part_count(path: str | Path, chunk_bytes: int = CHUNK_BYTES) -> int:
    if chunk_bytes < 1:
        raise ValueError('chunk_bytes must be positive')
    return math.ceil(Path(path).stat().st_size / chunk_bytes)


def read_part(path: str | Path, index: int, chunk_bytes: int = CHUNK_BYTES) -> bytes:
    """Read only requested part; index zero based."""
    count = part_count(path, chunk_bytes)
    if index < 0 or index >= count:
        raise IndexError(f'part index {index} out of range 0..{count - 1}')
    with Path(path).open('rb') as handle:
        handle.seek(index * chunk_bytes)
        return handle.read(chunk_bytes)


def part_filename(path: str | Path, index: int) -> str:
    return f'{Path(path).name}.part{index+1:03d}'


def windows_reassemble_batch(path: str | Path, expected_hash: str, chunk_bytes: int = CHUNK_BYTES) -> bytes:
    """Return a small .bat file, to be saved beside downloaded parts on Windows."""
    name = Path(path).name
    if any(c in name for c in '"&|<>^%!'):
        raise ValueError('Unsafe output filename for Windows batch script')
    names = [part_filename(path, i) for i in range(part_count(path, chunk_bytes))]
    lines = [
        '@echo off', 'setlocal', 'cd /d "%~dp0"',
        'echo Checking downloaded parts...',
    ]
    for n in names:
        lines += [f'if not exist "{n}" (', f'  echo Missing file: {n}', '  pause', '  exit /b 1', ')']
    joined = '+'.join(f'"{n}"' for n in names)
    lines += [
        'echo Combining parts...',
        f'copy /b {joined} "{name}" >nul',
        'if errorlevel 1 (',
        '  echo Failed to combine parts.', '  pause', '  exit /b 1', ')',
        'echo Verifying SHA256...',
        f'certutil -hashfile "{name}" SHA256',
        f'echo Expected SHA256: {expected_hash}',
        'echo Compare the hash printed above with the expected hash.',
        'echo If they are identical, the resulting file is complete.',
        'pause',
    ]
    return ('\r\n'.join(lines) + '\r\n').encode('utf-8')
