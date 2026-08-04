import os

# Must run before importing datasets/huggingface_hub: these timeout constants are read
# from the env at import time. The 10s default times out slow reads and spams backoff
# retries; raise it.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

import argparse  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from itertools import islice  # noqa: E402

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402
from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

SAVE_DIR: str = "./fleurs_files"
# A modest worker count is intentional: one stream per language saturates the link and
# causes read timeouts, so cap it instead of letting ThreadPoolExecutor pick min(32, cpu+4).
N_WORKERS: int = 4

FLEURS_LANGUAGES: list[str] = [
    "ar_eg",  # Arabic (Egypt)
    "cmn_hans_cn",  # Mandarin Chinese (Simplified)
    "de_de",  # German
    "el_gr",  # Greek
    "en_us",  # English
    "es_419",  # Spanish (Latin America)
    "fr_fr",  # French
    "id_id",  # Indonesian
    "it_it",  # Italian
    "ja_jp",  # Japanese
    "ko_kr",  # Korean
    "pt_br",  # Portuguese (Brazil)
    "ru_ru",  # Russian
    "ta_in",  # Tamil
    "th_th",  # Thai
    "ur_pk",  # Urdu
    "vi_vn",  # Vietnamese
]


def download_one_language(lang: str, num_samples: int) -> int:
    """Stream one language into 16 kHz WAV files under SAVE_DIR; skip existing. Returns new-file count."""
    # Skip the network entirely when this language is already complete.
    if all(
        os.path.isfile(os.path.join(SAVE_DIR, f"{lang}_{i:04d}.wav")) for i in range(num_samples)
    ):
        return 0

    new_count = 0
    try:
        dataset = load_dataset(
            "google/fleurs",
            lang,
            split="validation",
            trust_remote_code=True,
            streaming=True,
        )
        for i, sample in enumerate(islice(dataset, num_samples)):
            wav_path = os.path.join(SAVE_DIR, f"{lang}_{i:04d}.wav")
            if os.path.isfile(wav_path):
                continue

            audio = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            sf.write(wav_path, audio, 16000)
            new_count += 1
    except Exception as e:
        tqdm.write(f"Error downloading {lang}: {e}")
    return new_count


def download_fleurs_data(num_samples: int = 20) -> int:
    """Download FLEURS audio into SAVE_DIR (flat, idempotent, concurrent). Returns new-file count."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    total_new = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(download_one_language, lang, num_samples): lang for lang in FLEURS_LANGUAGES
        }
        with tqdm(total=len(FLEURS_LANGUAGES), desc="Languages", unit="lang") as pbar:
            for future in as_completed(futures):
                pbar.set_postfix(lang=futures[future])
                total_new += future.result()
                pbar.update(1)

    return total_new


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FLEURS calibration audio (16 kHz WAV).")
    parser.add_argument("--num-samples", type=int, default=20, help="samples per language")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    new_count = download_fleurs_data(args.num_samples)
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
