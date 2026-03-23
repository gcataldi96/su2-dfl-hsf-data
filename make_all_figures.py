from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURE_SCRIPTS = ("data_analysis_MAIN.py", "data_analysis_SM.py")


def main() -> None:
    data_dir = ROOT / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            "Missing data/ directory. This repository expects the canonical .npz datasets "
            "to be present under data/."
        )

    cache_dir = ROOT / ".cache"
    mplconfig_dir = cache_dir / "matplotlib"
    cache_dir.mkdir(exist_ok=True)
    mplconfig_dir.mkdir(exist_ok=True)
    child_env = dict(os.environ)
    child_env.setdefault("XDG_CACHE_HOME", str(cache_dir))
    child_env.setdefault("MPLCONFIGDIR", str(mplconfig_dir))

    for script_name in FIGURE_SCRIPTS:
        subprocess.run([sys.executable, str(ROOT / script_name)], check=True, env=child_env)


if __name__ == "__main__":
    main()
