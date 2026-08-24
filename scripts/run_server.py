import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "services" / "api" / "src"))

import uvicorn
from main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8085"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Launching BioSCADA AI Server on http://127.0.0.1:{port}/studio (host='{host}') ...")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
