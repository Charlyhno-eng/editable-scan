# SPDX-License-Identifier: MPL-2.0
"""Deterministic raster restoration without redrawing text or changing layout."""

import math
import zlib

import cv2
import numpy as np
import pikepdf
from PIL import Image

MAX_ENHANCED_PIXELS = 12_000_000


def _subpixel_ink(image):
    """Find LCD text fringes on mostly white pages; leave colored artwork alone."""
    if image.mode != "RGB" or image.width * image.height > 4_000_000:
        return None
    pixels = np.asarray(image)
    red, green, blue = cv2.split(pixels)
    low = np.minimum(np.minimum(red, green), blue)
    high = np.maximum(np.maximum(red, green), blue)
    if np.mean(low > 235) < 0.55:
        return None
    near = cv2.dilate((high < 170).astype("uint8"), np.ones((3, 5), "uint8")) > 0
    count = np.count_nonzero(near)
    if count < 100 or np.count_nonzero((high - low > 20) & near) < count * 0.1:
        return None
    r, b = red.astype("float32"), blue.astype("float32")
    mask = near[:, 1:-1]
    if not mask.any():
        return None
    same = np.abs(r[:, 1:-1] - b[:, 1:-1])[mask].mean()
    left = np.abs(r[:, 1:-1] - b[:, :-2])[mask].mean()
    right = np.abs(r[:, 1:-1] - b[:, 2:])[mask].mean()
    if min(left, right) >= same * 0.95:
        return None
    # Infer the sampling direction before excluding saturated artwork from the mask.
    near &= high.astype("int16") - low < 200
    return pixels if left < right else pixels[..., ::-1], near


def enhance_image(image):
    """Reconstruct subpixel edges when available, then upscale and sharpen ink.

    The original geometry, color artwork and formula shapes supply the result;
    no generative model invents characters. Transparent and bilevel images are
    kept intact. Upscaling is bounded independently of the input pixel limit.
    """
    if image.mode in ("P", "LA"):
        image = image.convert("RGBA")
    if image.mode == "RGBA":
        if image.getchannel("A").getextrema() != (255, 255):
            return image
        image = image.convert("RGB")
    if image.mode not in ("RGB", "L"):
        return image
    extrema = image.getextrema()
    if image.mode == "L":
        extrema = [extrema]
    if all(lo == hi for lo, hi in extrema):
        return image
    scale = max(
        1.0, min(3.0, math.sqrt(MAX_ENHANCED_PIXELS / (image.width * image.height)))
    )
    size = (
        max(image.width, int(image.width * scale)),
        max(image.height, int(image.height * scale)),
    )
    subpixel = _subpixel_ink(image)
    enlarged = (
        image.resize(size, Image.Resampling.LANCZOS)
        if size != image.size
        else image.copy()
    )
    if subpixel is not None:
        pixels, mask = subpixel
        # RGB subpixels carry three real horizontal samples of monochrome ink.
        samples = Image.fromarray(
            np.ascontiguousarray(pixels).reshape(image.height, image.width * 3)
        )
        ink = samples.resize(size, Image.Resampling.LANCZOS).convert("RGB")
        mask = Image.fromarray(mask.astype("uint8") * 255).resize(
            size, Image.Resampling.NEAREST
        )
        enlarged = Image.composite(ink, enlarged, mask)
    pixels = np.asarray(enlarged)
    blurred = cv2.GaussianBlur(pixels, (0, 0), max(0.6, scale * 0.35))
    sharp = cv2.addWeighted(pixels, 2.5, blurred, -1.5, 0)
    mask = cv2.compare(cv2.absdiff(pixels, blurred), 3, cv2.CMP_GT)
    result = pixels.copy()
    cv2.copyTo(sharp, mask, result)
    return Image.fromarray(result)


def _restore_samples(image):
    restored = enhance_image(image)
    if restored is image:
        return None
    return restored.mode, restored.size, zlib.compress(restored.tobytes(), level=1)


def prepare_pdf_images(page, pool):
    """Decode on the owning thread; restore pixels concurrently with page OCR."""
    jobs = []
    objects, seen = [], set()
    resources = [page.Resources]
    while resources:
        for obj in resources.pop().get("/XObject", {}).values():
            if obj.objgen in seen:
                continue
            seen.add(obj.objgen)
            if obj.get("/Subtype") == "/Image":
                objects.append(obj)
            elif obj.get("/Subtype") == "/Form":
                resources.append(obj.get("/Resources", {}))
    masks = {
        obj.SMask.objgen
        for obj in objects
        if isinstance(obj, pikepdf.Stream)
        and isinstance(obj.get("/SMask"), pikepdf.Stream)
    }
    for obj in objects:
        if not isinstance(obj, pikepdf.Stream) or obj.get("/Subtype") != "/Image":
            continue
        colorspace = obj.get("/ColorSpace")
        indexed = (
            isinstance(colorspace, pikepdf.Array)
            and len(colorspace) >= 2
            and colorspace[0] == "/Indexed"
            and colorspace[1] in (pikepdf.Name.DeviceRGB, pikepdf.Name.DeviceGray)
        )
        if (
            obj.get("/EditableScanEnhanced", False)
            or obj.objgen in masks
            or obj.get("/ImageMask", False)
            or "/SMask" in obj
            or "/Mask" in obj
            or int(obj.get("/BitsPerComponent", 0)) != 8
            or int(obj.Width) * int(obj.Height) > 40_000_000
            or not (
                indexed
                or colorspace in (pikepdf.Name.DeviceRGB, pikepdf.Name.DeviceGray)
            )
        ):
            continue
        # Never queue decoded rasters for an entire long document in memory.
        if len(jobs) == 2:
            finish_pdf_images([jobs.pop(0)])
        original = pikepdf.PdfImage(obj).as_pil_image()
        jobs.append((obj, pool.submit(_restore_samples, original)))
    return jobs


def finish_pdf_images(jobs):
    """Apply completed rasters on the owning thread; keep all PDF page objects."""
    for obj, future in jobs:
        restored = future.result()
        if restored is None:
            continue
        mode, size, data = restored
        obj.write(data, filter=pikepdf.Name.FlateDecode)
        for key in ("/Decode", "/DecodeParms"):
            if key in obj:
                del obj[key]
        obj.Width, obj.Height = size
        obj.ColorSpace = (
            pikepdf.Name.DeviceGray if mode == "L" else pikepdf.Name.DeviceRGB
        )
        obj.Interpolate = False
        obj.EditableScanEnhanced = True
