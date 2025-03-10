#!/usr/bin/env python3
"""
Main entry point for the RAG document converter.
"""

import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from rag_converter.cli import main

if __name__ == "__main__":
    sys.exit(main())
