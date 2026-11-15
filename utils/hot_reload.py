"""
Reload the dashboard's own modules when their files change on disk.

Streamlit Cloud pulls new commits into a running app, but Python keeps every module it has
already imported. A page that imports a NEW module (fresh code) which calls a function added to
an ALREADY-IMPORTED module then crashes with AttributeError until the app is rebooted, and pages
quietly keep showing old screens. refresh() is called at the top of main.py on every run: if any
loaded utils module's file is newer than when the module was first seen, all loaded utils
modules are reloaded (helpers first, views last, twice so cross-imports settle).
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from typing import Dict, List, Tuple

_SEEN: Dict[str, float] = {}          # module name -> wall-clock time it was first seen loaded (or last reloaded)


def _loaded(prefixes: Tuple[str, ...]) -> List[str]:
    out = []
    for name, mod in list(sys.modules.items()):
        if name == __name__ or not name.startswith(prefixes) or mod is None:
            continue
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".py") and os.path.exists(f):
            out.append(name)
    return out


def stale(prefixes: Tuple[str, ...] = ("utils.",)) -> List[str]:
    """Loaded modules whose file changed after the module was first seen. Registers unseen modules."""
    now, out = time.time(), []
    for name in _loaded(prefixes):
        if name not in _SEEN:
            _SEEN[name] = now
        elif os.path.getmtime(sys.modules[name].__file__) > _SEEN[name]:
            out.append(name)
    return out


def refresh(prefixes: Tuple[str, ...] = ("utils.",)) -> List[str]:
    """Reload every loaded module under `prefixes` if any of them is stale. Returns the stale names."""
    changed = stale(prefixes)
    if not changed:
        return []
    order = sorted(_loaded(prefixes), key=lambda n: (n.endswith("_view"), n))
    for _ in range(2):
        for name in order:
            try:
                importlib.reload(sys.modules[name])
            except Exception as e:                       # one bad module must not take the app down
                print(f"[hot_reload] {name}: {type(e).__name__}: {e}", flush=True)
    now = time.time()
    for name in order:
        _SEEN[name] = now
    print(f"[hot_reload] reloaded {len(order)} modules after changes in: {', '.join(changed[:8])}", flush=True)
    return changed
