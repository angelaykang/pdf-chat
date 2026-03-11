"""
Download Mistral 7B Instruct GGUF (Q4_K_M) into models/.

Usage:
  python download_model.py

The file is ~4.4 GB; download can take several minutes. Requires internet.
After this, run: python app_p1.py open_source
"""

import os
import sys
import time
import urllib.request

# Hugging Face direct link (no login)
URL = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)
DEST_NAME = "mistral-7b-instruct.Q4_K_M.gguf"
# Expected size in bytes (~4.37 GB) for a quick sanity check after download
EXPECTED_MIN_BYTES = 4_000_000_000
MAX_RETRIES = 3
RETRY_DELAY_SEC = 10


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    dest_path = os.path.join(models_dir, DEST_NAME)

    os.makedirs(models_dir, exist_ok=True)

    if os.path.isfile(dest_path):
        size = os.path.getsize(dest_path)
        if size >= EXPECTED_MIN_BYTES:
            print("Model already present: %s (%.1f GB)" % (dest_path, size / (1024**3)))
            return
        print("Existing file too small (%.1f MB). Re-downloading." % (size / (1024**2)))
        try:
            os.remove(dest_path)
        except OSError as e:
            print("Could not remove existing file: %s" % e)
            sys.exit(1)

    def progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100.0, 100.0 * done / total_size)
        done_mb = done / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print("\r  %.1f%% (%.0f / %.0f MB)" % (pct, done_mb, total_mb), end="", flush=True)

    print("Downloading Mistral 7B Instruct Q4_K_M (~4.4 GB).")
    print("This may take several minutes depending on your connection.")
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                print("\nRetry %d/%d in %d seconds..." % (attempt, MAX_RETRIES, RETRY_DELAY_SEC))
                time.sleep(RETRY_DELAY_SEC)
            urllib.request.urlretrieve(URL, dest_path, reporthook=progress)
            print()  # newline after progress
            break
        except Exception as e:
            last_error = e
            print("\nDownload attempt %d failed: %s" % (attempt, e))
            if os.path.isfile(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
    else:
        print("Download failed after %d attempts." % MAX_RETRIES)
        if last_error:
            print("Last error: %s" % last_error)
        print("\nManual download:")
        print("  1. Open in browser: %s" % URL)
        print("  2. Save the file as: %s" % dest_path)
        sys.exit(1)

    size = os.path.getsize(dest_path)
    if size < EXPECTED_MIN_BYTES:
        print(
            "Warning: Downloaded file is smaller than expected (%.1f MB). "
            "It may be incomplete or the model may have changed." % (size / (1024**2))
        )
    else:
        print("Saved to: %s (%.1f GB)" % (dest_path, size / (1024**3)))
    print("\nNext step: python app_p1.py open_source")


if __name__ == "__main__":
    main()
