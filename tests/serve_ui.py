"""Serve the front end with mock backends and an isolated, temporary session store."""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from xcortex.config import Settings
from xcortex.server.app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8418)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="xcortex-ui-") as directory:
        root = Path(directory)
        settings = Settings()
        settings.app.mode = "mock"
        settings.paths.data = str(root / "data")
        settings.paths.db = str(root / "db")
        settings.paths.artifacts = str(root / "artifacts")
        settings.paths.state = str(root / "state.json")
        uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
