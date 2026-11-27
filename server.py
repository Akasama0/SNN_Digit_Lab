"""Serve the digit verifier from the computer running the classifier."""
import argparse
import base64
import binascii
import hmac
import io
import json
import secrets
import socket
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image, UnidentifiedImageError

from test_new_images import prepare_image, read_weights
from train import encode_input_spikes, spike_counts

ROOT = Path(__file__).resolve().parent
LIMIT = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 30_000_000


class Handler(BaseHTTPRequestHandler):
    def send_data(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, data):
        self.send_data(status, json.dumps(data).encode(), "application/json")

    def do_GET(self):
        if urlsplit(self.path).path == "/":
            self.send_data(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_json(404, {"error": "Page not found."})

    def do_POST(self):
        route = urlsplit(self.path)
        if route.path != "/classify":
            self.send_json(404, {"error": "Page not found."})
            return
        if not hmac.compare_digest(self.headers.get("X-Digit-Key", "").encode(), self.server.key.encode()):
            self.send_json(403, {"error": "Open the full phone URL printed on the computer."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= LIMIT:
            self.send_json(400, {"error": "Choose a photo under 20 MB."})
            return
        query = parse_qs(route.query)
        expected = query.get("expected", [""])[0]
        if expected not in ("", "0", "1"):
            self.send_json(400, {"error": "Expected digit must be 0 or 1."})
            return
        try:
            self.connection.settimeout(30)
            photo = self.rfile.read(length)
            if len(photo) != length:
                raise ValueError("Photo transfer ended early.")
            with Image.open(io.BytesIO(photo), formats=("JPEG", "PNG", "WEBP")) as source:
                image_format = source.format.lower()
                if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError("Choose a photo under 30 megapixels.")
            canvas, pixels = prepare_image(io.BytesIO(photo), query.get("light") == ["1"])
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, binascii.Error):
            self.send_json(422, {"error": "Use a clear JPEG, PNG, or WebP photo under 30 megapixels with a visible digit. For HEIC, try the phone camera in the page or choose JPEG."})
            return

        n0, n1 = map(int, spike_counts(encode_input_spikes(pixels[None, :]), *self.server.model)[0])
        prediction = int(n1 > n0)
        result = {
            "id": uuid.uuid4().hex,
            "prediction": prediction,
            "spikes": [n0, n1],
            "tie": n0 == n1,
            "expected": int(expected) if expected else None,
            "match": prediction == int(expected) if expected else None,
            "pixels": pixels.tolist(),
        }
        small = Image.fromarray(pixels.astype("uint8").reshape(8, 8))
        try:
            with tempfile.TemporaryDirectory(dir=self.server.output) as directory:
                capture = Path(directory) / result["id"]
                capture.mkdir()
                (capture / f"original.{image_format}").write_bytes(photo)
                canvas.save(capture / "input_28x28.png")
                small.save(capture / "input_8x8.png")
                (capture / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                (capture / "rtl_vectors.txt").write_text(
                    " ".join(map(str, [*pixels, n0, n1, prediction])) + "\n", encoding="ascii"
                )
                capture.rename(self.server.output / result["id"])
        except OSError:
            self.send_json(500, {"error": "The computer could not save the capture. Check the output folder."})
            return
        preview = io.BytesIO()
        small.save(preview, format="PNG")
        result["preview"] = "data:image/png;base64," + base64.b64encode(preview.getvalue()).decode("ascii")
        self.send_json(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=ROOT / "captures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        server.key = secrets.token_urlsafe(18)
        server.model = read_weights(ROOT / "digit_weights.vh")
        server.output = args.output.resolve()
        hosts = ["127.0.0.1"] if args.host == "0.0.0.0" else [args.host]
        if args.host == "0.0.0.0":
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                    route.connect(("192.0.2.1", 80))
                    address = route.getsockname()[0]
                if address not in hosts:
                    hosts.append(address)
            except OSError:
                pass
        for index, host in enumerate(hosts):
            note = "This is for computer" if index == 0 else "this url is for your phone"
            print(f"Open http://{host}:{server.server_port}/#{server.key}  ({note})", flush=True)
        print(f"Saving photos to {server.output}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
