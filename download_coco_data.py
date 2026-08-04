import argparse
import asyncio
import os
import random

import aiofiles
import aiohttp
from datasets import load_dataset
from tqdm.asyncio import tqdm

SAVE_DIR: str = "./coco_files"
DATASET: str = "ChristophSchuhmann/MS_COCO_2017_URL_TEXT"
SEED: int = 42
CONCURRENCY_LIMIT: int = 50  # simultaneous downloads; too high (e.g. 500) risks server blocks


async def download_image(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> int:
    """Download one image into SAVE_DIR. Returns 1 on success, 0 on failure."""
    async with semaphore:
        filepath = os.path.join(SAVE_DIR, os.path.basename(url.split("?")[0]))
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()
                async with aiofiles.open(filepath, mode="wb") as f:
                    async for chunk in response.content.iter_chunked(1024):
                        await f.write(chunk)
            return 1
        except Exception:
            return 0


async def download_coco_images(num_samples: int) -> int:
    """Download num_samples random MS COCO images into SAVE_DIR (concurrent). Returns success count."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    urls = load_dataset(DATASET, split="train")["URL"]
    random.seed(SEED)
    target_urls = random.sample(urls, num_samples)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url, semaphore) for url in target_urls]
        results = await tqdm.gather(*tasks)

    return sum(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MS COCO calibration images.")
    parser.add_argument("--num-samples", type=int, default=100, help="number of images to download")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = asyncio.run(download_coco_images(args.num_samples))
    print(f"Done: {count}/{args.num_samples} images in {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
