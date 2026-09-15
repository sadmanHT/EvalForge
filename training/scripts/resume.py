from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).with_name("train.py")
    if "--resume-from-checkpoint" not in sys.argv[1:]:
        raise SystemExit("resume.py requires --resume-from-checkpoint <checkpoint>")
    env = dict(os.environ)
    return subprocess.call([sys.executable, str(script), *sys.argv[1:]], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
