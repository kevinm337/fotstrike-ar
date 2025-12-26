from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

app = FastAPI(title="FotStrike AR API", version="0.2.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    frame_id = 0
    try:
        while True:
            await asyncio.sleep(0.1)  # 100ms
            await ws.send_json({"frame_id": frame_id, "players": []})
            frame_id += 1
    except WebSocketDisconnect:
        # client closed the connection
        pass
    except Exception:
        # any unexpected error; try to close cleanly
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Run: python -m app.main
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
