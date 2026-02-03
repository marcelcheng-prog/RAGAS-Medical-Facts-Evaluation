"""
Medical Facts Evaluation entry point.

Usage:
    python -m medical_facts_evaluation [options]
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
