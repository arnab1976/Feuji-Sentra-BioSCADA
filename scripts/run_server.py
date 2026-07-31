import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "services" / "api" / "src"))

import uvicorn
from main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    host = os.getenv("HOST", "::")
    print(f"Launching BioSCADA AI Server on http://localhost:{port}/studio (Dual-stack host='{host}') ...")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
