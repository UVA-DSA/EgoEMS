#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import re
import argparse
import spacy

# Path to the pre-generated beep
BEEP_PATH = "./data/beep.wav"

def to_seconds(ts: str) -> float:
    parts = list(map(float, ts.split(':')))
    if len(parts) == 3:
        h, m, s = parts
        return h*3600 + m*60 + s
    m, s = parts
    return m*60 + s

def process_pair(json_path, wav_path, nlp):
    base, _ = os.path.splitext(json_path)
    out_json  = base + "_deidentified.json"
    out_audio = base + "_deidentified.mp3"

    redaction_main_log = "./data/redaction_main.log"
    os.makedirs(os.path.dirname(redaction_main_log), exist_ok=True)

    with open(redaction_main_log, "a") as log_file:
        log_file.write(f"Processing: {json_path} → {out_json}\n")

    print(f"\n[+] Processing:\n    {json_path}\n    {wav_path}")
    print(f"Output    → {out_json}\n    → {out_audio}")

    # check if output files already exist
    if os.path.exists(out_json) and os.path.exists(out_audio):
        print(f"[-] Output files already exist: {out_json} and {out_audio}")
        return

    # 1) load transcript
    with open(json_path, 'r') as f:
        transcript = json.load(f)

    # 2) flatten words & collect tokens
    flat = []
    tokens = []
    for utt in transcript:
        for w in utt["Words"]:
            clean = w["word"].strip('.,?!')
            tokens.append(clean)
            flat.append({
                "text": clean,
                "start": to_seconds(w["start"]),
                "end":   to_seconds(w["end"]),
                "word_obj": w,
                "utt_obj": utt
            })

    # 3) load YOUR sensitive terms list
    sensitive_file = "./data/sensitive_entities.txt"
    if os.path.exists(sensitive_file):
        with open(sensitive_file) as f:
            sensitive = {line.strip().lower() for line in f if line.strip()}
        print(f"[+] Loaded {len(sensitive)} sensitive terms")
    else:
        print(f"[-] No sensitive_terms file at {sensitive_file}; nothing will be redacted.")
        sensitive = set()

    # 4) redact JSON + collect bleep segments
    bleep_segments = []
    for item in flat:
        txt_lower = item["text"].lower()
        if txt_lower in sensitive:
            start, end = item["start"], item["end"]
            bleep_segments.append((start, end))
            item["word_obj"]["word"] = "[REDACTED]"
            patt = r"\b" + re.escape(item["text"]) + r"\b"
            item["utt_obj"]["Utterance"] = re.sub(
                patt, "[REDACTED]",
                item["utt_obj"]["Utterance"],
                flags=re.IGNORECASE
            )
            print(f"    [*] Redacted '{item['text']}' at {start:.3f}-{end:.3f}")
            with open(redaction_main_log, "a") as log_file:
                log_file.write(f"Redacted '{item['text']}' at {start:.3f}-{end:.3f}\n")

    # 5) save redacted JSON
    with open(out_json, "w") as f:
        json.dump(transcript, f, indent=2)
    print(f"[+] Wrote redacted JSON → {out_json}")

    # 6) if no bleeps, done
    if not bleep_segments:
        print("[-] No PII spans to bleep.")
        # save original audio as MP3 with the output name
        cmd = [
            "ffmpeg", "-y", "-i", wav_path,
            "-c:a", "libmp3lame",
            "-b:a", "64k",
            "-ar", "16000",
            out_audio
        ]
        print(f"[+] Running ffmpeg to copy original audio to {out_audio}…")
        subprocess.run(cmd, check=True)
        print(f"[+] Wrote original audio → {out_audio}")
        print(f"[*] No PII spans found; original audio saved as MP3.")
        return

    # 7) build ffmpeg filter
    # a) mute original at all spans
    vol_chain = ",".join(
        f"volume=enable='between(t,{s-0.2},{e+0.2})':volume=0"
        for s, e in bleep_segments
    )
    # b) load & delay the single pre-generated beep for each segment
    beep_parts = []
    for i, (s, e) in enumerate(bleep_segments):
        ms = int(s * 1000)
        beep_parts.append(f"amovie={BEEP_PATH},adelay={ms}|{ms}[bd{i}];")
    # c) mix original+bleeps
    inputs = "[orig]" + "".join(f"[bd{i}]" for i in range(len(bleep_segments)))
    mix = f"{inputs}amix=inputs={len(bleep_segments)+1}:duration=longest[out]"

    filter_complex = f"[0]{vol_chain}[orig];" + "".join(beep_parts) + mix

    # 8) run ffmpeg: compress to 64 kbps mono 16 kHz MP3
    cmd = [
        "ffmpeg", "-y", "-i", wav_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "libmp3lame",
        "-b:a", "64k",
        "-ar", "16000",
        out_audio
    ]
    print(f"[+] Running ffmpeg to bleep {len(bleep_segments)} segment(s)…")
    subprocess.run(cmd, check=True)
    print(f"[+] Wrote censored audio → {out_audio}")

def main(root_dir):
    # load SpaCy once
    print("[*] Loading SpaCy en_core_web_trf…")
    # nlp = spacy.load("en_core_web_trf")
    nlp = None

    # ensure data directory exists, and pre-generate beep if needed
    os.makedirs(os.path.dirname(BEEP_PATH), exist_ok=True)
    if not os.path.exists(BEEP_PATH):
        print(f"[*] Generating beep file at {BEEP_PATH}…")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", "sine=frequency=1000:duration=0.5",
             BEEP_PATH],
            check=True
        )
    print(f"[+] Using beep → {BEEP_PATH}")

    # walk recursively
    for dirpath, _, files in os.walk(root_dir):
        if os.path.basename(dirpath).lower() != "audio":
            continue
        jsons = [f for f in files if f.lower().endswith(".json")]
        # wavs  = {f for f in files if f.lower().endswith(".wav")} # for cars and opvrs
        wavs  = {f for f in files if f.lower().endswith(".mp3")}
        print("-=" * 20)
        print("\n[*] Found audio folder:", dirpath)
        for jf in jsons:
            json_name = os.path.splitext(jf)[0]
            if json_name.endswith("_gemini_timestamped"):
                json_name = json_name[:-len("_gemini_timestamped")]
            else:
                continue  # skip non-gemini files
            # wav_name = json_name + ".wav" # for cars and opvrs
            wav_name = json_name + ".mp3" 
            if wav_name in wavs:
                process_pair(
                    os.path.join(dirpath, jf),
                    os.path.join(dirpath, wav_name),
                    nlp
                )
                print(f"Processing pair: {jf} and {wav_name}")
        # only process the first "audio" folder per directory
        # break
        print("-=" * 20)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Recursively censor PII in all audio/json pairs under a root directory"
    )
    p.add_argument("root_dir", help="Top-level folder to recurse into")
    args = p.parse_args()
    main(args.root_dir)
