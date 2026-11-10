"""Project-relative paths; independent of the current working directory."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / 'test_data'
OUTPUT_ROOT = PROJECT_ROOT / 'results'
