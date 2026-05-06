#!/usr/bin/env python3

import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, Request

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
app = FastAPI()
logs = []


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/log")
async def log(request: Request):
    data = await request.json()
    record = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "client": request.client.host if request.client else None,
        "data": data,
    }
    logs.append(record)
    logging.info("%s", record)
    return {"ok": True}


def _main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    import uvicorn

    uvicorn.run("server:app", host=host, port=port)


if __name__ == "__main__":
    _main()