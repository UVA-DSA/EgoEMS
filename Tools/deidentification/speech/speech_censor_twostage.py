#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import re
import argparse
from typing import List, Tuple

# -------- Config --------
BEEP_FREQ_HZ = 1000.0
BEEP_PAD_S   = 0.20      # pad before/after each span
MIN_BLEEP_S  = 0.15      # drop spans shorter than this after padding
OUT_SR       = 16000
OUT_CH       = 1
OUT_BR       = "64k"

SENSITIVE_PATH = "./data/sensitive_entities.txt"
REDACTION_LOG  = "./data/redaction_main.log"

# ------------------------

def to_seconds(ts: str) -> float:
    parts = list(map(float, ts.split(':')))
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    m, s = parts
    return m * 60 + s

def merge_and_clean_spans(spans: List[Tuple[float, float]],
                          pad: float,
                          min_len: float) -> List[Tuple[float, float]]:
    """
    Expand each (s,e) by pad on both ends, drop invalid/zero spans,
    merge overlaps/adjacent intervals, and enforce min duration.
    """
    cleaned = []
    for s, e in spans:
        if e < s:
            # swap if inverted, or drop if identical after swap still tiny
            s, e = e, s
        s = max(0.0, s - pad)
        e = max(s, e + pad)
        if (e - s) <= 1e-6:
            continue
        cleaned.append((s, e))

    if not cleaned:
        return []

    # sort and merge
    cleaned.sort()
    merged = [cleaned[0]]
    for s, e in cleaned[1:]:
        ls, le = merged[-1]
        if s <= le + 1e-6:  # overlap/adjacent
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))

    # enforce minimum duration
    result = [(s, e) for (s, e) in merged if (e - s) >= min_len]
    return result

def process_pair(json_path, wav_path):
    base, _ = os.path.splitext(json_path)
    out_json  = base + "_deidentified.json"
    out_audio = base + "_deidentified.mp3"

    os.makedirs(os.path.dirname(REDACTION_LOG), exist_ok=True)
    with open(REDACTION_LOG, "a") as log_file:
        log_file.write(f"Processing: {json_path} → {out_json}\n")

    print(f"\n[+] Processing:\n    {json_path}\n    {wav_path}")
    print(f"Output    → {out_json}\n    → {out_audio}")

    if os.path.exists(out_json) and os.path.exists(out_audio):
        print(f"[-] Output files already exist: {out_json} and {out_audio}")
        return

    # 1) load transcript
    with open(json_path, 'r') as f:
        transcript = json.load(f)

    # 2) flatten words & collect tokens
    flat = []
    for utt in transcript:
        for w in utt.get("Words", []):
            clean = w["word"].strip('.,?!')
            flat.append({
                "text": clean,
                "start": to_seconds(w["start"]),
                "end":   to_seconds(w["end"]),
                "word_obj": w,
                "utt_obj": utt
            })

    # 3) sensitive terms
    if os.path.exists(SENSITIVE_PATH):
        with open(SENSITIVE_PATH) as f:
            sensitive = {line.strip().lower() for line in f if line.strip()}
        print(f"[+] Loaded {len(sensitive)} sensitive terms")
    else:
        print(f"[-] No sensitive_terms file at {SENSITIVE_PATH}; nothing will be redacted.")
        sensitive = set()

    # 4) redact JSON + raw spans
    raw_spans = []
    for item in flat:
        txt_lower = item["text"].lower()
        if txt_lower in sensitive:
            s, e = item["start"], item["end"]
            raw_spans.append((s, e))
            item["word_obj"]["word"] = "[REDACTED]"
            patt = r"\b" + re.escape(item["text"]) + r"\b"
            item["utt_obj"]["Utterance"] = re.sub(
                patt, "[REDACTED]",
                item["utt_obj"]["Utterance"],
                flags=re.IGNORECASE
            )
            print(f"    [*] Redacted '{item['text']}' at {s:.3f}-{e:.3f}")
            with open(REDACTION_LOG, "a") as log_file:
                log_file.write(f"Redacted '{item['text']}' at {s:.3f}-{e:.3f}\n")

    # 5) save redacted JSON
    with open(out_json, "w") as f:
        json.dump(transcript, f, indent=2)
    print(f"[+] Wrote redacted JSON → {out_json}")

    # 6) clean spans
    spans = merge_and_clean_spans(raw_spans, pad=BEEP_PAD_S, min_len=MIN_BLEEP_S)

    if not spans:
        print("[-] No valid PII spans to bleep (after cleaning).")
        # copy original → MP3 mono 16k
        cmd = [
            "ffmpeg", "-y", "-i", wav_path,
            "-ac", str(OUT_CH),
            "-ar", str(OUT_SR),
            "-c:a", "libmp3lame",
            "-b:a", OUT_BR,
            out_audio
        ]
        print(f"[+] Running ffmpeg to copy original audio to {out_audio}…")
        subprocess.run(cmd, check=True)
        print(f"[+] Wrote original audio → {out_audio}")
        return

    # 7) build filter_complex (robust)
    #
    # Strategy:
    #   a) Feed [0:a] → force mono@16k → chain volume mutes over all spans → [orig]
    #   b) For each span (s,e), synthesize a sine tone of duration (e-s)
    #      at mono@16k, then delay by s (ms). Label each [bd{i}]
    #   c) Sum all beep branches → [beeps] with normalize=0
    #   d) Mix [orig] + [beeps] → [mix] with normalize=0, then soft-limit → [out]
    #
    # This avoids amovie/EOF quirks and prevents amix level shrink.
    #
    volume_chain_labels = []
    last_label = "a0"
    filter_parts = []
    # input -> format
    filter_parts.append(
        f"[0:a]aformat=sample_rates={OUT_SR}:channel_layouts=mono[{last_label}]"
    )

    # chained mutes
    for idx, (s, e) in enumerate(spans):
        s = max(0.0, s)
        e = max(s, e)
        nxt = f"a{idx+1}"
        # use precise times in enable; chain label to label
        filter_parts.append(
            f"[{last_label}]volume=enable='between(t,{s:.6f},{e:.6f})':volume=0[{nxt}]"
        )
        last_label = nxt
        volume_chain_labels.append(last_label)

    # final orig label
    filter_parts.append(f"[{last_label}]anull[orig]")

    # beep branches from sine sources
    for i, (s, e) in enumerate(spans):
        dur = max(MIN_BLEEP_S, e - s)
        start_ms = int(round(s * 1000.0))
        # `sine` keeps everything in-graph, set SR and mono explicitly
        filter_parts.append(
            f"sine=frequency={BEEP_FREQ_HZ}:sample_rate={OUT_SR}:duration={dur:.6f},"
            f"aformat=sample_rates={OUT_SR}:channel_layouts=mono,"
            f"adelay={start_ms}:all=1[bd{i}]"
        )

    # sum all beep branches
    if len(spans) == 1:
        filter_parts.append(f"[bd0]anull[beeps]")
    else:
        beep_inputs = "".join(f"[bd{i}]" for i in range(len(spans)))
        filter_parts.append(
            f"{beep_inputs}amix=inputs={len(spans)}:duration=longest:normalize=0[beeps]"
        )

    # final 2-input mix + soft limiting to avoid clips
    filter_parts.append(
        "[orig][beeps]amix=inputs=2:duration=longest:normalize=0,"
        "alimiter=limit=0.95[out]"
    )

    filter_complex = ";".join(filter_parts)

    # 8) run ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ac", str(OUT_CH),
        "-ar", str(OUT_SR),
        "-c:a", "libmp3lame",
        "-b:a", OUT_BR,
        out_audio
    ]

    print(f"[+] Running ffmpeg with {len(spans)} merged span(s)…")
    # For debugging, uncomment:
    # print("FILTER:", filter_complex)
    subprocess.run(cmd, check=True)
    print(f"[+] Wrote censored audio → {out_audio}")

def main(root_dir):
    for dirpath, _, files in os.walk(root_dir):
        if os.path.basename(dirpath).lower() != "audio":
            continue
        jsons = [f for f in files if f.lower().endswith(".json")]
        wavs  = {f for f in files if f.lower().endswith(".wav")}
        print("-=" * 20)
        print("\n[*] Found audio folder:", dirpath)
        for jf in jsons:
            name = os.path.splitext(jf)[0]
            if name.endswith("_gemini_timestamped"):
                stem = name[: -len("_gemini_timestamped")]
            else:
                continue
            wav_name = stem + ".wav"
            if wav_name in wavs:
                process_pair(
                    os.path.join(dirpath, jf),
                    os.path.join(dirpath, wav_name),
                )
                print(f"Processing pair: {jf} and {wav_name}")
            else:
                print(f"[-] No matching audio for {jf}: looking for {wav_name}")
        print("-=" * 20)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Recursively censor PII in all audio/json pairs under a root directory"
    )
    p.add_argument("root_dir", help="Top-level folder to recurse into")
    args = p.parse_args()
    main(args.root_dir)
