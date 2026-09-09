"""
HAT-Light Satellite Super-Resolution Studio - Application Entry Point
Zero-config launcher: automatically checks Python environment, starts the backend, and opens the studio.

Usage:
    python app.py
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def find_python_interpreter() -> str:
    # 1. Local virtual environment in repo
    local_venv = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if local_venv.exists():
        return str(local_venv)
    
    # 2. Parent directory virtual environment
    parent_venv = REPO_ROOT.parent / ".venv" / "Scripts" / "python.exe"
    if parent_venv.exists():
        return str(parent_venv)

    # 3. Fallback to current sys.executable
    return sys.executable

def main():
    print("=" * 70)
    print("  HAT-Light: 4-Channel Satellite Super-Resolution Studio")
    print("=" * 70)
    py_exec = find_python_interpreter()
    print(f"[1/3] Using Python interpreter: {py_exec}")

    # Launch Studio Backend
    server_script = REPO_ROOT / "studio" / "backend" / "server.py"
    print(f"[2/3] Launching Inference Server ({server_script.name})...")
    
    server_process = subprocess.Popen(
        [py_exec, str(server_script)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Wait for server initialization
    print("[3/3] Initializing studio and loading neural network weights...")
    time.sleep(3.5)

    studio_url = "http://localhost:8000"
    print(f"\n   Studio live at: {studio_url}")
    print("Opening browser for interactive exploration...")
    webbrowser.open(studio_url)

    print("\nPress Ctrl+C to stop the studio.")
    try:
        while True:
            line = server_process.stdout.readline()
            if line:
                print(line.strip())
            elif server_process.poll() is not None:
                break
    except KeyboardInterrupt:
        print("\nShutting down studio...")
        server_process.terminate()

if __name__ == "__main__":
    main()
