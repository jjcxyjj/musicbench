"""Run after activating/installing the MusicBench harness environment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from musicbench.setup_metrics import main

if __name__ == '__main__':
    raise SystemExit(main())
