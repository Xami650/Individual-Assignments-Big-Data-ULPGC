import sys
from pathlib import Path

# Make python/matmul.py importable without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
