"""python -m localtts  ->  serve the API (the systemd unit runs exactly this)."""

import logging

import uvicorn

from .config import settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("localtts.server:app", host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
