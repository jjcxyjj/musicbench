"""Reproducibility metadata collection (seed, versions, code commit, model hash)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _try_module_version(name: str) -> Optional[str]:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import torch  # type: ignore
        info["torch_version"] = torch.__version__
        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:
            info["cuda_version"] = torch.version.cuda
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_version"] = None
    return info


def hash_file(path: str, chunk_size: int = 1 << 20) -> Optional[str]:
    """MD5 of a file, or None if the path does not exist."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_dir(path: str) -> Optional[str]:
    """Deterministic MD5 over a directory's file names + contents (sorted)."""
    if not path or not os.path.isdir(path):
        return None
    h = hashlib.md5()
    for root, _dirs, files in os.walk(path):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, path)
            h.update(rel.encode("utf-8"))
            fh = hash_file(fp)
            if fh:
                h.update(fh.encode("utf-8"))
    return h.hexdigest()


def collect_env(seed: Optional[int], model_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the ``env.json`` payload recorded next to every run."""
    cuda = _cuda_info()
    return {
        "seed": seed,
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "torch": {
            "version": cuda.get("torch_version"),
        },
        "cuda": {
            "available": cuda.get("available", False),
            "version": cuda.get("cuda_version"),
            "device_name": cuda.get("device_name"),
        },
        "numpy_version": _try_module_version("numpy"),
        "scipy_version": _try_module_version("scipy"),
        "sklearn_version": _try_module_version("sklearn"),
        "mir_eval_version": _try_module_version("mir_eval"),
        "code_commit": _git_commit(),
        "model": model_meta or {},
    }


def dumps_jsonable(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
