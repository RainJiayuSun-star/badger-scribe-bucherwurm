#!/usr/bin/env python3
"""Run Badger Scribe pages through UW's remote Churro-3B service.

This compatibility entry point deliberately stays small. The implementation is
split into pipeline modules so a future local Churro backend can share the
same dataset, logging, prompt, and orchestration code.
"""

import sys
from pathlib import Path

# `python pipeline/transcribe_gateway.py` puts pipeline/ rather than the
# project root on sys.path. Add the root so the package imports work without
# requiring installation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
