import argparse
import datetime as dt
import glob
import json
import os
import re
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

SEED_PATH = "../conversation_seed_data.json"   # shared seed
CATEGORY = "tool_call"
MAX_SEQLEN = 2048       # truncate the final transcript (matches prepare_calibset)
MAX_NEW_TOKENS = 512    # cap per generated turn
TOOLCALL_QUOTA = 16     # base tool_call samples (recipe distribution); scaled by --ratio

# Same model tags as prepare_calibset.py.
MODEL_IDS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]


def parse_tool_calls_from_text(text: str) -> List[Dict[str, object]]:
    """Extract <tool_call>{...}</tool_call> blocks (Qwen/Hermes convention) into tool_call dicts."""
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
    parsed_calls: List[Dict[str, object]] = []
    for index, payload in enumerate(matches, start=1):
        try:
            data = json.loads(payload.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        parsed_calls.append(
            {
                "id": str(data.get("id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                },
            }
        )
    return parsed_calls


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_date(value: str):
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _parse_utc_offset(value: str):
    normalized = value.strip().upper().replace("UTC", "")
    match = re.fullmatch(r"([+-])(\d{2}):(\d{2})", normalized)
    if not match:
        return None
    sign = 1 if match.group(1) == "+" else -1
    hours = int(match.group(2))
    minutes = int(match.group(3))
    delta = dt.timedelta(hours=hours, minutes=minutes) * sign
    return dt.timezone(delta)


def execute_local_tool_call(name: str, arguments: Dict[str, object]) -> Dict[str, object]:
    """Deterministic canned tool results (no network)."""
    if name == "get_weather_forecast":
        location = str(arguments.get("location", "Unknown"))
        days = max(1, min(_to_int(arguments.get("days"), 2), 7))
        start_date_raw = str(arguments.get("start_date", dt.date.today().isoformat()))
        start_date = _parse_iso_date(start_date_raw) or dt.date.today()
        forecast = []
        conditions = ["Sunny", "Partly cloudy", "Light rain", "Clear"]
        for offset in range(days):
            current_date = start_date + dt.timedelta(days=offset)
            condition = conditions[offset % len(conditions)]
            forecast.append(
                {
                    "date": current_date.isoformat(),
                    "condition": condition,
                    "high_c": 19 + offset,
                    "low_c": 11 + offset,
                }
            )
        summary = "Comfortable daytime temperatures with a mix of clear and lightly cloudy periods."
        if any(item["condition"] == "Light rain" for item in forecast):
            summary = "Mostly mild weather with a chance of light rain during part of the period."
        return {"location": location, "forecast": forecast, "summary": summary}

    if name == "get_exchange_rate":
        base = str(arguments.get("base", "USD")).upper()
        quote = str(arguments.get("quote", "KRW")).upper()
        rates = {("USD", "KRW"): 1382.5, ("EUR", "USD"): 1.08, ("JPY", "USD"): 0.0067}
        rate = rates.get((base, quote), 1.2345)
        return {"base": base, "quote": quote, "rate": rate, "as_of": "2026-03-31"}

    if name == "get_asset_price":
        symbol = str(arguments.get("symbol", arguments.get("asset", "UNKNOWN"))).upper()
        prices = {
            "AAPL": {"price": 214.32, "currency": "USD", "change_pct_24h": 1.2},
            "NVDA": {"price": 942.15, "currency": "USD", "change_pct_24h": 2.8},
            "TSLA": {"price": 188.76, "currency": "USD", "change_pct_24h": -0.9},
            "BTC": {"price": 84250.0, "currency": "USD", "change_pct_24h": 3.6},
            "ETH": {"price": 4680.0, "currency": "USD", "change_pct_24h": 2.1},
            "SPX": {"price": 5288.37, "currency": "USD", "change_pct_24h": 0.6},
            "IXIC": {"price": 16642.21, "currency": "USD", "change_pct_24h": 0.9},
        }
        selected = prices.get(symbol, {"price": 123.45, "currency": "USD", "change_pct_24h": 0.0})
        return {"symbol": symbol, **selected, "as_of": "2026-03-31"}

    if name == "get_sports_schedule":
        team = str(arguments.get("team", "Team"))
        opponent = str(arguments.get("opponent", "Opponent"))
        league = str(arguments.get("league", "league")).upper()
        return {
            "league": league,
            "team": team,
            "games": [
                {"date": "2026-04-02", "opponent": opponent if opponent != "Opponent" else "Wolves"},
                {"date": "2026-04-05", "opponent": "Kings"},
            ],
        }

    if name == "get_sports_standings":
        league = str(arguments.get("league", "league")).upper()
        division = str(arguments.get("division", "division"))
        return {
            "league": league,
            "division": division,
            "standings": [
                {"team": "Team A", "wins": 11, "losses": 3, "points": 22},
                {"team": "Team B", "wins": 9, "losses": 5, "points": 18},
                {"team": "Team C", "wins": 8, "losses": 6, "points": 16},
            ],
        }

    if name == "get_time_by_timezone":
        offsets = arguments.get("utc_offsets")
        if isinstance(offsets, list) and offsets:
            requested_offsets = [str(item) for item in offsets]
        else:
            requested_offsets = [
                str(arguments.get("timezone", arguments.get("utc_offset", arguments.get("offset", "+00:00"))))
            ]
        now_utc = dt.datetime.now(dt.timezone.utc)
        resolved = []
        for raw_offset in requested_offsets:
            normalized = raw_offset if raw_offset.upper().startswith("UTC") else f"UTC{raw_offset}"
            tz = _parse_utc_offset(normalized) or dt.timezone.utc
            resolved.append(
                {"timezone": normalized, "local_time": now_utc.astimezone(tz).isoformat(timespec="seconds")}
            )
        return resolved[0] if len(resolved) == 1 else {"times": resolved}

    return {"tool": name, "arguments": arguments, "result": f"Result for {name} is available."}


def load_seed():
    """Return (tool_call seed conversations, tool schemas) from the shared seed JSON."""
    with open(SEED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    conversations = data["seed_conversations_by_lang"][CATEGORY]
    tools = data["tool_schemas_by_lang"][CATEGORY]
    return conversations, tools


def generate_sample(model, tokenizer, device, seed_conv, tools):
    """Self-generate one tool-calling conversation: model calls a tool, local result is injected, model finalizes.

    Returns (final transcript token ids, whether a tool call actually occurred).
    """
    messages: List[Dict[str, object]] = [
        {"role": "system", "content": str(seed_conv["system"])},
        {"role": "user", "content": str(seed_conv["initial_user"])},
    ]
    tool_injected = False
    used_tool = False

    while True:
        input_ids = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        assistant_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

        parsed = parse_tool_calls_from_text(assistant_text) if not tool_injected else []
        if parsed:
            used_tool = True
            messages.append({"role": "assistant", "content": "", "tool_calls": parsed})
            for call in parsed:
                fn = call["function"]
                fn_name = str(fn["name"])
                try:
                    fn_args = json.loads(str(fn["arguments"]))
                except json.JSONDecodeError:
                    fn_args = {}
                result = execute_local_tool_call(fn_name, fn_args)
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
                        "name": fn_name,
                        "tool_call_id": str(call["id"]),
                    }
                )
            tool_injected = True
            continue  # re-prompt so the model produces the final answer from the tool result

        messages.append({"role": "assistant", "content": assistant_text})
        break

    token_ids = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=True)
    return token_ids[:MAX_SEQLEN], used_tool


def update_calib_manifest(save_dir: str, category: str) -> str:
    """Add this category's .npy paths to the manifest that prepare_calibset.py created.

    prepare_calibset.py is run first and *creates* calib_files.json (wiki languages).
    This *modifies* it: it drops any stale entries for `category` (so re-runs are
    idempotent) and appends the freshly generated ones, leaving the language entries
    untouched. Requires the manifest to already exist.
    """
    manifest_path = os.path.join(save_dir, "calib_files.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found. Run prepare_calibset.py first; "
            f"generate_tool-call-data.py only adds '{category}' to an existing calibset."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    tag = f"{os.sep}{category}{os.sep}"
    kept = [entry for entry in manifest["calib paths"] if tag not in entry[0]]
    category_npys = sorted(glob.glob(os.path.join(save_dir, category, "*.npy")))
    kept += [[os.path.abspath(p)] for p in category_npys]
    manifest["calib paths"] = kept

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-generate tool_call calibration data (inputs_embeds) for a Qwen3 model.")
    parser.add_argument("--model", required=True, choices=MODEL_IDS, help="model id to generate tool_call calib for")
    parser.add_argument("--ratio", type=float, default=1.0, help="scale the tool_call quota (1.0 = recipe distribution)")
    parser.add_argument("--seed", type=int, default=42, help="base seed; sample i uses seed+i for reproducible variety")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id = args.model
    num_samples = round(TOOLCALL_QUOTA * args.ratio)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").eval().to(device)
    embedding_weight = model.get_input_embeddings().weight.detach()

    conversations, tools = load_seed()

    name = model_id.split("/")[-1]
    save_dir = f"./calibset_{name}"
    out_dir = os.path.join(save_dir, CATEGORY)
    text_dir = os.path.join(out_dir, "texts")
    os.makedirs(text_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, "inputs_embeds_*.npy")):  # exact count on re-run
        os.remove(stale)

    hits = 0
    for i in tqdm(range(num_samples), desc=f"{name}/{CATEGORY}", unit="sample"):
        set_seed(args.seed + i)  # reproducible, yet distinct when the same seed conversation is reused
        seed_conv = conversations[i % len(conversations)]
        token_ids, used_tool = generate_sample(model, tokenizer, device, seed_conv, tools)
        hits += int(used_tool)

        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=embedding_weight.device)
        inputs_embeds = embedding_weight[token_tensor].unsqueeze(0).float().cpu().numpy()  # [1, seq, dim]
        np.save(os.path.join(out_dir, f"inputs_embeds_{i}.npy"), inputs_embeds)

        with open(os.path.join(text_dir, f"sample_{i:05d}.txt"), "w", encoding="utf-8") as f:
            f.write(tokenizer.decode(token_ids, skip_special_tokens=False))

    manifest = update_calib_manifest(save_dir, CATEGORY)
    print(f"Done: {num_samples} {CATEGORY} samples ({hits} with a real tool call) in {os.path.abspath(out_dir)}")
    print(f"Manifest: {os.path.abspath(manifest)}")


if __name__ == "__main__":
    main()
