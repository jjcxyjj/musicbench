"""Standalone local inference; run from any working directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ACE.inference import cli
if __name__ == '__main__':
    cli('transcriber')
