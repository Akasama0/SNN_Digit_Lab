"""Prepare new digit photos and test the all-LIF Nano 9K classifier."""
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from train import encode_input_spikes, spike_counts

ROOT = Path(__file__).resolve().parent
EDGES = tuple(i * 28 // 8 for i in range(9))


def prepare_image(path, light_on_dark):
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
    ink = gray if light_on_dark else ImageOps.invert(gray)
    if ink.getextrema()[0] == ink.getextrema()[1]:
        raise ValueError(f"{path} has no brightness variation")
    ink = ImageOps.autocontrast(ink)
    ink = ink.point([0 if level < 64 else level for level in range(256)])
    bounds = ink.getbbox()
    if bounds is None:
        raise ValueError(f"{path} has no visible stroke")
    digit = ink.crop(bounds)
    scale = 20 / max(digit.size)
    size = tuple(max(1, round(length * scale)) for length in digit.size)
    digit = digit.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(digit, ((28 - size[0]) // 2, (28 - size[1]) // 2))

    array = np.asarray(canvas)
    pooled = np.array([
        int(array[top:bottom, left:right].sum()) // ((bottom - top) * (right - left))
        for top, bottom in zip(EDGES, EDGES[1:])
        for left, right in zip(EDGES, EDGES[1:])
    ], dtype=np.int32)
    return canvas, pooled


def signed_hex(token, bits):
    number = int(token, 16)
    return number - (1 << bits) if number >= (1 << (bits - 1)) else number


def read_weights(path):
    source = path.read_text(encoding="ascii")
    bias = re.search(r"MODEL_BIAS\s*=\s*24'sh([0-9a-fA-F]{6})", source)
    shift = re.search(r"SCORE_SHIFT\s*=\s*(\d+)", source)
    slots = re.findall(r"6'd(\d+):\s*weight_at\s*=\s*8'sh([0-9a-fA-F]{2})", source)
    if bias is None or shift is None or not slots:
        raise ValueError(f"Cannot read the model in {path}")
    weights = np.zeros(64, dtype=np.int32)
    for slot, value in slots:
        weights[int(slot)] = signed_hex(value, 8)
    return weights, signed_hex(bias.group(1), 24), int(shift.group(1))


def write_demo_rom(path, features):
    # If given one image, let both buttons select it.
    images = features if len(features) == 2 else np.repeat(features, 2, axis=0)
    lines = ["// Prepared by test_new_images.py.",
             "function [7:0] sample_at;", "    input [6:0] address;",
             "    begin", "        case (address)"]
    lines.extend(f"            7'd{i}: sample_at = 8'd{value};"
                 for i, value in enumerate(images.ravel()) if value)
    lines.extend(["            default: sample_at = 8'd0;", "        endcase",
                  "    end", "endfunction"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="One or two JPG/PNG files")
    parser.add_argument("--light-on-dark", action="store_true",
                        help="Use for bright strokes on a dark background")
    parser.add_argument("--out", type=Path, default=Path("image_test"))
    args = parser.parse_args()
    if len(args.images) > 2:
        parser.error("Provide one or two images")
    destination = args.out.resolve()
    if destination == ROOT:
        parser.error("Use an output directory separate from the Verilog source folder")
    try:
        prepared = [prepare_image(path, args.light_on_dark) for path in args.images]
        weights, bias, shift = read_weights(ROOT / "digit_weights.vh")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    features = np.stack([pixels for _, pixels in prepared])
    counts = spike_counts(encode_input_spikes(features), weights, bias, shift)
    destination.mkdir(parents=True, exist_ok=True)
    vectors = []
    for i, ((canvas, pixels), (n0, n1)) in enumerate(zip(prepared, counts)):
        canvas.save(destination / f"input_{i}_28x28.png")
        image8 = Image.fromarray(pixels.astype(np.uint8).reshape(8, 8))
        image8.resize((256, 256), Image.Resampling.NEAREST).save(
            destination / f"input_{i}_8x8_preview.png")
        prediction = int(n1 > n0)
        tie = " (tie)" if n0 == n1 else ""
        print(f"{args.images[i]} -> {prediction}; spikes: 0={n0}, 1={n1}{tie}")
        vectors.append(" ".join(map(str, [*pixels, n0, n1, prediction])))

    write_demo_rom(destination / "demo_images.vh", features)
    (destination / "rtl_vectors.txt").write_text("\n".join(vectors) + "\n", encoding="ascii")
    print(f"Previews, simulation vectors, and button ROM saved to {destination}")


if __name__ == "__main__":
    main()

