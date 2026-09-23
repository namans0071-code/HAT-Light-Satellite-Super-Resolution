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

    # Wait for server initialization with active health check
    print("[3/3] Initializing studio and loading neural network weights...")
    studio_url = "http://127.0.0.1:8000"
    health_url = f"{studio_url}/api/health"

    import urllib.request
    server_ready = False
    start_time = time.time()
    max_wait = 45  # allow up to 45s for model weights to load

    while time.time() - start_time < max_wait:
        if server_process.poll() is not None:
            # Server exited unexpectedly
            out, _ = server_process.communicate()
            print("\n" + "=" * 70)
            print("ERROR: Inference Server failed to start! Output:")
            print(out)
            print("=" * 70)
            return

        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "StudioLauncher"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    server_ready = True
                    break
        except Exception:
            pass

        print(".", end="", flush=True)
        time.sleep(0.5)

    print()
    if server_ready:
        print(f"\n   >>> Studio live and healthy at: {studio_url} <<<")
        print("Opening browser for interactive exploration...")
        webbrowser.open(studio_url)
    else:
        print(f"\n[Warning] Studio server took longer than expected to respond, opening {studio_url} anyway...")
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
