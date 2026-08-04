import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

from datasets import load_dataset
from tqdm import tqdm

SAVE_DIR: str = "./wikipedia_files"
SNAPSHOT: str = "20231101"

WIKIPEDIA_LANGUAGES: list[str] = [
    "en",  # English
    "de",  # German
    "fr",  # French
    "es",  # Spanish
    "it",  # Italian
    "ja",  # Japanese
    "ko",  # Korean
    "zh",  # Chinese
    "ru",  # Russian
    "pt",  # Portuguese
    "hi",  # Hindi
    "th",  # Thai
]


def download_one_language(lang: str, num_samples: int) -> int:
    """Stream one Wikipedia language into raw .txt files under SAVE_DIR; skip existing. Returns new-file count."""
    new_count = 0
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            f"{SNAPSHOT}.{lang}",
            split="train",
            streaming=True,
        )
        for i, row in enumerate(islice(dataset, num_samples)):
            txt_path = os.path.join(SAVE_DIR, f"{lang}_{i:04d}.txt")
            if os.path.isfile(txt_path):
                continue

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(row["text"])
            new_count += 1
    except Exception as e:
        tqdm.write(f"Error downloading {lang}: {e}")
    return new_count


def download_wikipedia_data(num_samples: int = 500) -> int:
    """Download Wikipedia articles into SAVE_DIR (raw .txt, flat, idempotent, concurrent). Returns new-file count."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    total_new = 0
    with ThreadPoolExecutor() as pool:
        futures = {
            pool.submit(download_one_language, lang, num_samples): lang for lang in WIKIPEDIA_LANGUAGES
        }
        with tqdm(total=len(WIKIPEDIA_LANGUAGES), desc="Languages", unit="lang") as pbar:
            for future in as_completed(futures):
                pbar.set_postfix(lang=futures[future])
                total_new += future.result()
                pbar.update(1)

    return total_new


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw Wikipedia text for calibration.")
    parser.add_argument("--num-samples", type=int, default=500, help="articles per language")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    new_count = download_wikipedia_data(args.num_samples)
    print(f"Done: {new_count} new files in {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()

    # Flush first: os._exit() skips Python finalization (and its buffer flush),
    # so buffered stdout would be lost when output is piped/redirected.
    # We skip finalization to avoid a PyGILState_Release crash from lingering
    # PyArrow/aiohttp streaming threads. See huggingface/datasets#7357.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
