from __future__ import annotations

import threading
import time
import webbrowser

import httpx
import uvicorn

from ..server.app import create_app


def _wait_for_health(url: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{url}/api/health", timeout=1.0)
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    return False


def launch(settings, browser: bool = False) -> None:
    app = create_app(settings)
    config = uvicorn.Config(
        app, host=settings.app.host, port=settings.app.port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://{settings.app.host}:{settings.app.port}"
    if not _wait_for_health(url):
        raise RuntimeError(f"server did not start at {url}")
    try:
        if browser:
            webbrowser.open(url)
            while thread.is_alive():
                time.sleep(0.5)
        else:
            import webview

            webview.create_window(
                "X-Cortex",
                url,
                width=1380,
                height=880,
                min_size=(1120, 720),
                background_color="#0b0d12",
            )
            webview.start()
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
