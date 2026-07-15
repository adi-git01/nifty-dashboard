"""
Atomic file writes.

Plain df.to_csv(path)/to_parquet(path) writes directly to the final path.
If the process dies mid-write (OOM, CI timeout, disk full, a killed job),
the file at `path` is left truncated — and unlike a hard crash, nothing
downstream necessarily notices: pandas can happily parse a truncated CSV
as a shorter, silently-wrong file. This is the same failure class that
produced the equity-curve phantom -27% loss and the Market Breadth/Mood
History zero-row corruption earlier — both were CSV history files being
appended to daily by a process that could be interrupted mid-write.

Writing to a temp file in the same directory and os.replace()-ing over the
real path is atomic on POSIX: readers only ever see the fully-written old
file or the fully-written new one, never a partial one.
"""
import os
import json


def atomic_to_csv(df, path, **kwargs):
    tmp_path = f"{path}.tmp{os.getpid()}"
    try:
        df.to_csv(tmp_path, **kwargs)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def atomic_json_dump(obj, path, **kwargs):
    """
    Same rationale as atomic_to_csv, for JSON state files. Some of these
    (the portfolio snapshot, positions.json) are read back with a
    parse-failure fallback that resets to an empty/default state — a
    truncated write from an interrupted process wouldn't just lose data,
    it would silently wipe live portfolio/position state on the next read.
    """
    tmp_path = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(obj, f, **kwargs)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise
