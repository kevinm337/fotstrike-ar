from __future__ import annotations

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="FotStrike AR API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Run: python -m app.main
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # turn off in production
        log_level="info",
    )
