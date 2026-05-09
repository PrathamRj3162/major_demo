"""
Test the deployed Render API with real images:
  1. A real human chest X-ray  → should be ACCEPTED
  2. A dog X-ray               → should be REJECTED

Usage:  python test_render_api.py
"""

import os
import sys
import io
import json
import urllib.request
import tempfile

RENDER_API = "https://major-demo-1.onrender.com/api"

# ── real test images ─────────────────────────────────────────────────────
# Human chest X-ray (Wikimedia Commons — PA view)
CHEST_XRAY_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/"
    "c/c8/Chest_Xray_PA_3-8-2010.png"
)

# Dog X-ray (Wikimedia Commons — lateral thorax)
DOG_XRAY_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/"
    "b/b0/Lateral_canine_xray.jpg"
)


def download_image(url, label):
    """Download image and return bytes + temp file path."""
    print(f"  ⬇  Downloading {label}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        print(f"  ✅ Downloaded {len(data) / 1024:.0f} KB")
        return data
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return None


def send_to_predict(image_bytes, filename):
    """
    POST the image to the /api/predict endpoint on Render.
    Uses multipart/form-data just like the frontend does.
    """
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
            status = resp.status
            data = json.loads(resp.read().decode("utf-8"))
            return status, data
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            data = json.loads(e.read().decode("utf-8"))
        except Exception:
            data = {"error": str(e)}
        return status, data
    except Exception as e:
        return None, {"error": str(e)}


def print_result(label, status, data, expected_accept):
    """Pretty-print the API response."""
    print(f"\n{'━' * 60}")
    print(f"  {label}")
    print(f"  Expected: {'ACCEPT ✓' if expected_accept else 'REJECT ✗'}")
    print(f"{'━' * 60}")
    print(f"  HTTP Status: {status}")

    is_accepted = data.get("is_chest_xray", None)

    # print validation checks if present
    checks = data.get("validation_checks", [])
    if checks:
        print(f"\n  Validation Checks:")
        for chk in checks:
            icon = "✅" if chk.get("passed") else "❌"
            print(f"    {icon} {chk['name']:25s} — {chk.get('detail', '')}")

    if is_accepted is True:
        pred = data.get("prediction", "N/A")
        conf = data.get("confidence", 0)
        review = data.get("needs_review", False)
        probs = data.get("probabilities", {})
        print(f"\n  ➤ VERDICT: ACCEPTED as chest X-ray")
        print(f"  ➤ Prediction: {pred} ({conf*100:.1f}% confidence)")
        print(f"  ➤ Probabilities: {probs}")
        if review:
            print(f"  ⚠  Flagged for manual review (low confidence)")
    elif is_accepted is False:
        reason = data.get("rejection_reason", data.get("error", "Unknown"))
        print(f"\n  ➤ VERDICT: REJECTED")
        print(f"  ➤ Reason: {reason}")
    else:
        print(f"\n  ➤ Response: {json.dumps(data, indent=2)[:500]}")

    match = (is_accepted == expected_accept)
    status_text = "PASS ✅" if match else "FAIL ⚠️"
    print(f"\n  ➤ TEST RESULT: {status_text}")
    return match


def main():
    print("=" * 60)
    print("  RENDER API — CHEST X-RAY VALIDATOR TEST")
    print(f"  Target: {RENDER_API}")
    print("=" * 60)

    # 1. health check first
    print("\n📡 Checking if Render service is alive...")
    try:
        req = urllib.request.Request(
            f"{RENDER_API}/health",
            headers={"User-Agent": "TestScript/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            health = json.loads(resp.read().decode("utf-8"))
            print(f"  ✅ Service is up: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"  ❌ Service is not responding: {e}")
        print("  The Render free tier may need a minute to spin up.")
        print("  Please wait and try again.")
        return 1

    # 2. download test images
    print("\n📥 Downloading test images...")
    chest_bytes = download_image(CHEST_XRAY_URL, "Human chest X-ray")
    dog_bytes = download_image(DOG_XRAY_URL, "Dog X-ray")

    if not chest_bytes:
        print("❌ Could not download chest X-ray — cannot continue.")
        return 1

    results = []

    # 3. test with human chest X-ray → should ACCEPT
    print("\n🔬 Sending human chest X-ray to API...")
    status, data = send_to_predict(chest_bytes, "chest_xray.png")
    passed = print_result("HUMAN CHEST X-RAY", status, data, expected_accept=True)
    results.append(("Human chest X-ray", passed))

    # 4. test with dog X-ray → should REJECT
    if dog_bytes:
        print("\n🔬 Sending dog X-ray to API...")
        status, data = send_to_predict(dog_bytes, "dog_xray.jpg")
        passed = print_result("DOG X-RAY", status, data, expected_accept=False)
        results.append(("Dog X-ray", passed))
    else:
        print("\n⚠  Skipping dog X-ray test (download failed)")

    # 5. summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    for name, ok in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
    print(f"\n  Total: {total}  |  Passed: {passed_count}  |  Failed: {total - passed_count}")
    print("=" * 60)

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
