#!/usr/bin/env python3
from run_stage import main

if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0], "--stage", "core"]
    raise SystemExit(main())

