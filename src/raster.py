# SPDX-License-Identifier: MPL-2.0
"""Compact PDF rasters with bounded precision and no JPEG artifacts on text."""

import io
import struct

import numpy as np
import pikepdf
from PIL import Image


def _indexed(image, colors):
    """Map an exact small palette without running a color quantizer."""
    pixels = np.asarray(image)
    keys = (
        (pixels[..., 0].astype("uint32") << 16)
        | (pixels[..., 1].astype("uint32") << 8)
        | pixels[..., 2]
    )
    lookup = np.empty(1 << 24, dtype="uint8")
    palette = []
    for index, (_, (red, green, blue)) in enumerate(colors):
        lookup[(red << 16) | (green << 8) | blue] = index
        palette.extend((red, green, blue))
    indexed = Image.fromarray(lookup[keys]).convert("P")
    indexed.putpalette(palette)
    return indexed


def encode_raster(image, *, compact=True):
    """Return encoded samples and PDF parameters, independently of a PDF owner.

    Document RGB samples use a 32-level grid: each channel changes by at most
    4/255, with black and white exact. No pixels, edges or colors are spatially
    averaged. ICC-managed and transparent images bypass this precision change.
    """
    profile = image.info.get("icc_profile", b"")
    if image.mode == "RGB":
        pixels = np.asarray(image)
        if (
            compact
            and np.array_equal(pixels[..., 0], pixels[..., 1])
            and np.array_equal(pixels[..., 1], pixels[..., 2])
            and not profile
        ):
            image = image.getchannel("R")
        elif compact and not profile and (colors := image.getcolors(256)) is not None:
            image = _indexed(image, colors)
        elif compact and not profile:
            # Restrict the small precision reduction to paper-like scans.
            sample = pixels[
                :: max(1, image.height // 400), :: max(1, image.width // 400)
            ]
            if np.mean(sample.min(axis=2) > 235) > 0.55:
                table = np.rint(
                    np.rint(np.arange(256) * (31 / 255)) * (255 / 31)
                ).astype("uint8")
                image = Image.fromarray(table[pixels])
    if image.mode == "CMYK":
        import zlib

        return (
            zlib.compress(image.tobytes(), 6),
            pikepdf.Name.FlateDecode,
            None,
            image.mode,
            8,
            None,
        )
    if (
        compact
        and image.mode == "RGB"
        and (colors := image.getcolors(256)) is not None
        and not profile
    ):
        image = _indexed(image, colors)
    buffer = io.BytesIO()
    image.save(buffer, "PNG", compress_level=6)
    data = buffer.getvalue()
    offset, chunks, palette = 8, [], None
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        if kind == b"IHDR":
            bits, color_type = payload[8:10]
        elif kind == b"IDAT":
            chunks.append(payload)
        elif kind == b"PLTE":
            palette = payload
        offset += length + 12
    colors = 3 if color_type == 2 else 1
    params = pikepdf.Dictionary(
        Predictor=15, Colors=colors, Columns=image.width, BitsPerComponent=bits
    )
    return b"".join(chunks), pikepdf.Name.FlateDecode, params, image.mode, bits, palette


def write_raster(obj, encoded, size, *, colorspace=None):
    data, filter_name, params, mode, bits, palette = encoded
    obj.write(data, filter=filter_name, decode_parms=params)
    for key in ("/Decode",):
        if key in obj:
            del obj[key]
    obj.Type = pikepdf.Name.XObject
    obj.Subtype = pikepdf.Name.Image
    obj.Width, obj.Height = size
    obj.BitsPerComponent = bits
    if mode == "P":
        colorspace = pikepdf.Array(
            [
                pikepdf.Name.Indexed,
                pikepdf.Name.DeviceRGB,
                len(palette) // 3 - 1,
                pikepdf.String(palette),
            ]
        )
    obj.ColorSpace = colorspace or {
        "L": pikepdf.Name.DeviceGray,
        "1": pikepdf.Name.DeviceGray,
        "RGB": pikepdf.Name.DeviceRGB,
        "CMYK": pikepdf.Name.DeviceCMYK,
    }.get(mode)
    obj.Interpolate = False
