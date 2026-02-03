#!/usr/bin/env python3
"""Debug .env loading"""
import os
from pathlib import Path
from dotenv import load_dotenv

print(f"CWD: {Path.cwd()}")
env_path = Path('.env')
print(f".env exists in CWD: {env_path.exists()}")

if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env from CWD")
else:
    print("No .env in CWD")

key = os.getenv("OPENAI_API_KEY", "")
print(f"OPENAI_API_KEY loaded: {bool(key)}")
if key:
    print(f"Key starts with: {key[:20]}...")
