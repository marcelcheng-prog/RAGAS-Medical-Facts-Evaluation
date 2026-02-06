"""
Medical Facts Evaluation entry point.

Usage:
    python -m medical_facts_evaluation [options]
"""

import warnings
# Suppress FutureWarning from google.generativeai deprecation in instructor
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress DeprecationWarnings from various libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
