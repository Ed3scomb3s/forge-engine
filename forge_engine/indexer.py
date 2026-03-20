from __future__ import annotations
import os
import json
import csv
import hashlib
from typing import Optional, Dict, List, Tuple


__all__ = [
    "get_index_dir",
    "get_index_path",
    "header_fingerprint",
    "ensure_index",
    "find_seek_offset",
]


def get_index_dir(data_dir: str) -> str:
    """
    Return the directory path where sidecar indexes should be stored.
    Example: <data_dir>/_index
    """
    return os.path.join(data_dir, "_index")


def get_index_path(data_dir: str, base_filename: str) -> str:
    """
    Return the full path to the index file for a given base CSV filename.
    Example: <data_dir>/_index/<base_filename>.idx
    """
    return os.path.join(get_index_dir(data_dir), f"{base_filename}.idx")


def header_fingerprint(header_text: str) -> str:
    """
    Compute a stable fingerprint of the CSV header row to validate index compatibility.
    """
    try:
        return hashlib.sha1(header_text.strip().encode("utf-8")).hexdigest()
    except Exception:
        return ""


def _read_header_text_binary(fb) -> str:
    """
    Read the first line (header) from a binary file object and return it decoded as UTF-8 (no trailing newline).
    """
    header_bytes = fb.readline()
    if not header_bytes:
        return ""
    try:
        return header_bytes.decode("utf-8").rstrip("\r\n")
    except Exception:
        # Fall back to latin-1 to avoid hard failure on odd encodings
        return header_bytes.decode("latin-1", errors="replace").rstrip("\r\n")


def _open_time_col_from_header(header_text: str, open_time_field: str = "open_time") -> int:
    """
    Parse header and return index of the open_time field. Defaults to 0 if not found.
    """
    try:
        cols = next(csv.reader([header_text]))
        name_to_idx = {str(h).strip(): i for i, h in enumerate(cols)}
        return int(name_to_idx.get(open_time_field, 0))
    except Exception:
        return 0


def _load_index_rows(idx_path: str) -> Tuple[Dict[str, int], List[Tuple[str, int]]]:
    """
    Load index file and return (meta, rows) where:
      - meta: dict of header JSON
      - rows: list of (open_time_iso, offset) tuples (sorted by time)
    """
    meta: Dict[str, int] = {}
    rows: List[Tuple[str, int]] = []
    with open(idx_path, "r", encoding="utf-8") as ix:
        header_line = ix.readline()
        try:
            meta = json.loads(header_line) if header_line else {}
        except Exception:
            meta = {}
        for line in ix:
            s = line.strip()
            if not s:
                continue
            # Format: <open_time_iso>,<offset>
            parts = s.rsplit(",", 1)
            if len(parts) != 2:
                continue
            t, ofs_s = parts
            try:
                ofs = int(ofs_s)
            except Exception:
                continue
            rows.append((t, ofs))
    return meta, rows


def _binary_search_time(rows: List[Tuple[str, int]], target_iso: str) -> Optional[int]:
    """
    Binary search smallest index i where rows[i].time >= target_iso. Return offset or None.
    """
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid][0] < target_iso:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(rows):
        return rows[lo][1]
    return None


def ensure_index(base_file: str, data_dir: str, open_time_field: str = "open_time") -> Dict[str, object]:
    """
    Ensure a sidecar index exists for base_file inside data_dir/_index.

    Index format:
      - First line: JSON meta
          {
            "base_filename": "...",
            "size": <file_size>,
            "mtime": <int(mtime)>,
            "header_sha1": "<sha1(header_text)>",
            "open_time_col": <int>
          }
      - Subsequent lines: "<open_time_iso>,<byte_offset>"

    Returns a dict:
      {
        "path": idx_path,
        "created": bool,          # True if built now
        "valid": bool,            # True if up-to-date with base_file
        "open_time_col": int      # Column index of the open_time field
      }
    """
    idx_dir = get_index_dir(data_dir)
    try:
        os.makedirs(idx_dir, exist_ok=True)
    except Exception:
        # Ignore directory creation failure; downstream open() will throw
        pass

    base_filename = os.path.basename(base_file)
    idx_path = get_index_path(data_dir, base_filename)

    # Stat base CSV
    try:
        st = os.stat(base_file)
    except Exception:
        return {"path": idx_path, "created": False, "valid": False, "open_time_col": 0}

    # Check if existing index is valid
    if os.path.exists(idx_path):
        try:
            with open(base_file, "rb") as fb:
                header_text = _read_header_text_binary(fb)
            hp = header_fingerprint(header_text)
            with open(idx_path, "r", encoding="utf-8") as ix:
                header_line = ix.readline()
                meta = json.loads(header_line) if header_line else {}
            if (
                int(meta.get("size", -1)) == st.st_size
                and int(meta.get("mtime", -1)) == int(st.st_mtime)
                and str(meta.get("base_filename")) == base_filename
                and str(meta.get("header_sha1", "")) == hp
            ):
                return {
                    "path": idx_path,
                    "created": False,
                    "valid": True,
                    "open_time_col": int(meta.get("open_time_col", 0)),
                }
        except Exception:
            # Fall through to rebuild
            pass

    # Build or rebuild index
    created = False
    try:
        with open(base_file, "rb") as fb, open(idx_path, "w", encoding="utf-8") as ixw:
            header_text = _read_header_text_binary(fb)
            if not header_text:
                return {"path": idx_path, "created": False, "valid": False, "open_time_col": 0}
            hp = header_fingerprint(header_text)
            i_ot = _open_time_col_from_header(header_text, open_time_field=open_time_field)

            # Write meta header
            meta = {
                "base_filename": base_filename,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "header_sha1": hp,
                "open_time_col": i_ot,
            }
            ixw.write(json.dumps(meta) + "\n")

            # Build lines: time,offset - read line by line, record offset before read
            while True:
                offset = fb.tell()
                line_bytes = fb.readline()
                if not line_bytes:
                    break
                try:
                    line = line_bytes.decode("utf-8").rstrip("\r\n")
                except Exception:
                    line = line_bytes.decode("latin-1", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                cols = line.split(",")
                if i_ot >= len(cols):
                    continue
                t = cols[i_ot].strip()
                if not t:
                    continue
                ixw.write(f"{t},{offset}\n")

            created = True
        return {"path": idx_path, "created": created, "valid": True, "open_time_col": i_ot}
    except Exception:
        # If build fails, mark invalid
        return {"path": idx_path, "created": False, "valid": False, "open_time_col": 0}


def find_seek_offset(index_path: str, target_iso: str) -> Optional[int]:
    """
    Return the byte offset into the CSV (body, not header) for the first row with open_time >= target_iso.
    Requires a valid index at index_path.
    """
    try:
        meta, rows = _load_index_rows(index_path)
        if not rows:
            return None
        return _binary_search_time(rows, target_iso)
    except Exception:
        return None