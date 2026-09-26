"""python -m localtts  ->  serve the API (the systemd unit runs exactly this)."""

import logging
import socket

import uvicorn

from .config import settings

log = logging.getLogger("localtts")

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _listen(host: str, port: int, family: socket.AddressFamily) -> socket.socket:
    s = socket.socket(family, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    s.bind((host, port))
    s.listen(2048)
    s.set_inheritable(True)
    return s


def _sockets() -> list[socket.socket]:
    """Loopback means both 127.0.0.1 and ::1: `localhost` resolves to ::1 first here, and a
    browser that tries it gets 'connection refused' if only IPv4 is listening."""
    if settings.host not in LOOPBACK:
        family = socket.AF_INET6 if ":" in settings.host else socket.AF_INET
        return [_listen(settings.host, settings.port, family)]
    socks = [_listen("127.0.0.1", settings.port, socket.AF_INET)]
    try:
        socks.append(_listen("::1", settings.port, socket.AF_INET6))
    except OSError as exc:  # IPv6 disabled: IPv4 alone still works
        log.warning("not listening on [::1]:%s (%s)", settings.port, exc)
    return socks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = uvicorn.Config(
        "localtts.server:app", log_level="info",
        # Browsers keep idle connections for minutes (Firefox 115 s, Chrome 300 s). uvicorn's
        # default of 5 s closes them first, and a request sent on a socket the server is closing
        # dies with a network error (browsers do not retry POSTs). Outlive the browser instead.
        timeout_keep_alive=settings.keep_alive_s,
    )
    server = uvicorn.Server(config)
    socks = _sockets()
    log.info("listening on %s", ", ".join(
        f"[{s.getsockname()[0]}]:{s.getsockname()[1]}" if s.family == socket.AF_INET6
        else f"{s.getsockname()[0]}:{s.getsockname()[1]}" for s in socks))
    server.run(sockets=socks)


if __name__ == "__main__":
    main()
