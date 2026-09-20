"""
PlacementPredict - Main Unified Application Launcher (app.py)
Run this single file to train (if needed), start the server, and launch the web UI:
    python app.py
"""

import os
import sys
import time
import webbrowser
import threading

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Configure UTF-8 output if possible
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import uvicorn

# Ensure artifacts directory and trained models exist before serving
def ensure_artifacts_ready():
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    linear_model = os.path.join(artifacts_dir, "linear_pipeline.joblib")
    tree_model = os.path.join(artifacts_dir, "tree_pipeline.joblib")
    metrics_file = os.path.join(artifacts_dir, "metrics.json")

    if not (os.path.exists(linear_model) and os.path.exists(tree_model) and os.path.exists(metrics_file)):
        print("[*] Pre-trained artifacts not found. Starting training lifecycle...")
        from src.train_all import run_full_training_lifecycle
        run_full_training_lifecycle(artifacts_dir)
    else:
        print("[*] Pre-trained models and artifacts found in 'artifacts/' folder.")


def open_browser():
    """Waits 1.5 seconds for the server to bind, then automatically opens the browser."""
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"\n[*] Opening PlacementPredict Web Dashboard in your default browser: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[-] Could not automatically open browser: {e}")


if __name__ == "__main__":
    print("=" * 65)
    print("⚡ PLACEMENTPREDICT: END-TO-END MACHINE LEARNING SYSTEM (M1, M2, M3)")
    print("=" * 65)
    
    # 1. Check & Ensure Models are Trained
    ensure_artifacts_ready()

    # 2. Start Browser in Background Thread
    threading.Thread(target=open_browser, daemon=True).start()

    # 3. Launch FastAPI Server
    print("\n[*] Starting FastAPI Server on http://127.0.0.1:8000 ...")
    print("[*] Press CTRL+C to stop the server.\n")
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
