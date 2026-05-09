"""
Test script for the Chest X-ray Validation pipeline.
Tests both locally AND against the deployed Render API.

Tests include:
  - Real human chest X-ray (Wikimedia)     → should be ACCEPTED
  - Synthetic dog-like X-ray               → should be REJECTED  
  - Colour photograph                      → should be REJECTED
  - Random grayscale noise                 → should be REJECTED
  - Panoramic image                        → should be REJECTED
  - All-white / All-black                  → should be REJECTED
  - Text document                          → should be REJECTED

Usage:
  python test_validator.py              # local model test
  python test_validator.py --render     # also test against Render API
"""

import sys
import os
import io
import json
import urllib.request
import traceback
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# make sure the backend package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import preprocess_image
from services.chest_xray_validator import validate_chest_xray

RENDER_API = "https://major-demo-1.onrender.com/api"

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_pil(arr):
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def _download_image(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data))
    except Exception as e:
        print(f"  ⚠  Could not download {url}: {e}")
        return None


# ── synthetic test image generators ──────────────────────────────────────────

def gen_synthetic_chest_xray():
    """Synthetic grayscale image resembling a chest X-ray."""
    size = 512
    img = np.zeros((size, size), dtype=np.float64)
    img += np.random.normal(20, 5, (size, size))
    yy, xx = np.ogrid[:size, :size]
    # lung fields
    left_lung = ((xx - 170)**2 / 120**2 + (yy - 260)**2 / 180**2) < 1
    right_lung = ((xx - 340)**2 / 120**2 + (yy - 260)**2 / 180**2) < 1
    img[left_lung] += 40
    img[right_lung] += 40
    # bright mediastinum
    centre = ((xx - 256)**2 / 60**2 + (yy - 256)**2 / 200**2) < 1
    img[centre] += 100
    # rib-like bands
    for y in range(100, 420, 40):
        img[y:y+6, :] += 60
    pil = _make_pil(img)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=2))
    return pil


def gen_dog_xray():
    """
    Synthetic grayscale image mimicking a dog/animal X-ray.
    Key differences from human chest X-ray:
    - Elongated horizontal body (quadruped, not upright)
    - 4 limb-like projections
    - Different organ placement
    This is grayscale and roughly square, so it should bypass
    the grayscale and aspect-ratio checks — testing whether the
    model-confidence checks catch it.
    """
    size = 512
    img = np.zeros((size, size), dtype=np.float64)
    img += np.random.normal(15, 3, (size, size))

    yy, xx = np.ogrid[:size, :size]

    # elongated horizontal body (quadruped torso)
    body = ((xx - 256)**2 / 200**2 + (yy - 240)**2 / 80**2) < 1
    img[body] += 80

    # spine — horizontal line through body centre
    img[235:245, 80:430] += 100

    # ribs — curved vertical lines (not horizontal like human)
    for x in range(120, 400, 30):
        for dy in range(-60, 60):
            y = 240 + dy
            if 0 <= y < size and 0 <= x < size:
                img[y, x] += 50

    # 4 legs (limb-like projections downward)
    for lx in [140, 220, 310, 390]:
        leg = ((xx - lx)**2 / 15**2 + (yy - 360)**2 / 80**2) < 1
        img[leg] += 60

    # skull at one end (circle)
    skull = ((xx - 460)**2 / 35**2 + (yy - 230)**2 / 40**2) < 1
    img[skull] += 90

    # tail at other end
    img[230:250, 30:80] += 40

    pil = _make_pil(img)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=2))
    return pil


def gen_colour_photo():
    arr = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    arr[50:200, 50:250] = [255, 80, 80]
    arr[100:300, 300:500] = [50, 200, 50]
    return _make_pil(arr)


def gen_panoramic_image():
    arr = np.random.randint(50, 200, (100, 800), dtype=np.uint8)
    return _make_pil(arr)


def gen_random_noise_grayscale():
    arr = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    return _make_pil(arr)


def gen_white_image():
    return Image.new("L", (300, 300), color=255)


def gen_black_image():
    return Image.new("L", (300, 300), color=0)


def gen_text_document():
    img = Image.new("L", (400, 400), color=240)
    draw = ImageDraw.Draw(img)
    for i in range(20):
        draw.text((10, 10 + i * 18), f"Line {i+1}: Lorem ipsum dolor sit amet", fill=30)
    return img


# ── Real image URLs ──────────────────────────────────────────────────────────

REAL_CHEST_XRAY_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/"
    "c/c8/Chest_Xray_PA_3-8-2010.png"
)

# ── local test runner ────────────────────────────────────────────────────────

def run_local_test(name, pil_image, expected_accept):
    """Run one image through the local validator."""
    print(f"\n{'─'*60}")
    print(f"  TEST: {name}")
    print(f"  Expected: {'ACCEPT ✓' if expected_accept else 'REJECT ✗'}")
    print(f"  Image size: {pil_image.size}, mode: {pil_image.mode}")
    print(f"{'─'*60}")

    try:
        tmp_path = os.path.join(os.path.dirname(__file__), "uploads", "__test_tmp.png")
        pil_image.save(tmp_path)
        tensor, pil_rgb = preprocess_image(tmp_path)
        result = validate_chest_xray(tensor, pil_rgb)

        for chk in result["checks"]:
            icon = "✅" if chk["passed"] else "❌"
            print(f"  {icon} {chk['name']:25s} — {chk['detail']}")

        accepted = result["is_chest_xray"]
        match = (accepted == expected_accept)

        if accepted:
            print(f"\n  ➤ VERDICT: ACCEPTED as chest X-ray")
        else:
            print(f"\n  ➤ VERDICT: REJECTED — {result.get('rejection_reason', 'N/A')}")

        status = "PASS ✅" if match else "FAIL ⚠️"
        print(f"  ➤ TEST RESULT: {status}")

        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return match

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        traceback.print_exc()
        return False


# ── Render API test ──────────────────────────────────────────────────────────

def send_to_render(image_bytes, filename):
    """POST image to the deployed Render /api/predict endpoint."""
    boundary = "----TestBoundary123456"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{RENDER_API}/predict",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "TestScript/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode("utf-8"))
        except Exception:
            data = {"error": str(e)}
        return e.code, data
    except Exception as e:
        return None, {"error": str(e)}


def run_render_test(name, pil_image, expected_accept):
    """Send an image to the Render API and check the result."""
    print(f"\n{'─'*60}")
    print(f"  RENDER TEST: {name}")
    print(f"  Expected: {'ACCEPT ✓' if expected_accept else 'REJECT ✗'}")
    print(f"{'─'*60}")

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    status, data = send_to_render(img_bytes, "test_image.png")
    print(f"  HTTP Status: {status}")

    checks = data.get("validation_checks", [])
    if checks:
        for chk in checks:
            icon = "✅" if chk.get("passed") else "❌"
            print(f"  {icon} {chk['name']:25s} — {chk.get('detail', '')}")

    is_accepted = data.get("is_chest_xray", None)

    if is_accepted is True:
        pred = data.get("prediction", "N/A")
        conf = data.get("confidence", 0)
        print(f"\n  ➤ VERDICT: ACCEPTED — {pred} ({conf*100:.1f}%)")
    elif is_accepted is False:
        reason = data.get("rejection_reason", data.get("error", "Unknown"))
        print(f"\n  ➤ VERDICT: REJECTED — {reason}")
    else:
        print(f"\n  ➤ Response: {json.dumps(data, indent=2)[:300]}")

    match = (is_accepted == expected_accept)
    print(f"  ➤ TEST RESULT: {'PASS ✅' if match else 'FAIL ⚠️'}")
    return match


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Also test against Render API")
    args = parser.parse_args()

    print("=" * 60)
    print("  CHEST X-RAY VALIDATOR — COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    # build test cases: (name, pil_image, expected_accept)
    tests = [
        ("Synthetic chest X-ray",      gen_synthetic_chest_xray(), True),
        ("Synthetic dog X-ray",        gen_dog_xray(),             False),
        ("Colour photograph",          gen_colour_photo(),         False),
        ("Panoramic grayscale",        gen_panoramic_image(),      False),
        ("Random grayscale noise",     gen_random_noise_grayscale(), False),
        ("All-white image",            gen_white_image(),          False),
        ("All-black image",            gen_black_image(),          False),
        ("Text document image",        gen_text_document(),        False),
    ]

    # download real chest X-ray
    print("\n📥 Downloading real chest X-ray...")
    real_cxr = _download_image(REAL_CHEST_XRAY_URL)
    if real_cxr:
        tests.insert(0, ("Real chest X-ray (Wikimedia)", real_cxr, True))

    # ── LOCAL tests ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LOCAL MODEL TESTS")
    print("=" * 60)

    local_results = []
    for name, pil_img, expected in tests:
        passed = run_local_test(name, pil_img, expected)
        local_results.append((name, passed))

    # ── RENDER tests (optional) ──────────────────────────────────────────
    render_results = []
    if args.render:
        print("\n\n" + "=" * 60)
        print("  RENDER API TESTS")
        print(f"  Target: {RENDER_API}")
        print("=" * 60)

        # wake up Render
        print("\n📡 Waking up Render service...")
        try:
            req = urllib.request.Request(f"{RENDER_API}/health", headers={"User-Agent": "Test/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                print(f"  ✅ Render is alive: {json.loads(resp.read())['status']}")
        except Exception as e:
            print(f"  ❌ Render not responding: {e}")
            print("  Skipping Render tests.")
            args.render = False

    if args.render:
        for name, pil_img, expected in tests:
            passed = run_render_test(name, pil_img, expected)
            render_results.append((name, passed))

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST SUMMARY — LOCAL MODEL")
    print("=" * 60)
    local_total = len(local_results)
    local_passed = sum(1 for _, p in local_results if p)
    for name, ok in local_results:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  Total: {local_total}  |  Passed: {local_passed}  |  Failed: {local_total - local_passed}")

    if render_results:
        print("\n" + "=" * 60)
        print("  TEST SUMMARY — RENDER API")
        print("=" * 60)
        render_total = len(render_results)
        render_passed = sum(1 for _, p in render_results if p)
        for name, ok in render_results:
            print(f"  {'✅' if ok else '❌'} {name}")
        print(f"\n  Total: {render_total}  |  Passed: {render_passed}  |  Failed: {render_total - render_passed}")

    all_passed = all(p for _, p in local_results) and all(p for _, p in render_results)
    if all_passed:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  SOME TESTS FAILED — Review validator thresholds.")

    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
