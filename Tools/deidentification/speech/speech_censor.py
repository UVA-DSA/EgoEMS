#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import re
import argparse
import spacy

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
    out_wav   = base + "_deidentified.wav"

    redaction_main_log = "./data/redaction_main.log"



    with open(redaction_main_log, "a") as log_file:
        log_file.write(f"Processing: {json_path} → {out_json}\n")

        print(f"\n[+] Processing:\n    {json_path}\n    {wav_path}")
        print(f"Output    → {out_json}\n    → {out_wav}")

        # 1) load transcript
        with open(json_path, 'r') as f:
            transcript = json.load(f)

        # 2) flatten words & collect tokens
        flat = []
        tokens = []
        for utt in transcript:
            for w in utt["Words"]:
                # print(f"    [*] Processing word: {w})")
                clean = w["word"].strip('.,?!')
                tokens.append(clean)
                flat.append({
                    "text": clean,
                    "start": to_seconds(w["start"]),
                    "end":   to_seconds(w["end"]),
                    "word_obj": w,
                    "utt_obj": utt
                })
        doc_text = " ".join(tokens)

        # 3) run NER
        doc = nlp(doc_text)
        sensitive = {ent.text for ent in doc.ents if ent.label_ in ("PERSON","GPE","ORG")}
        print(f"[+] Detected entities: {sensitive}")


        # append the entities to a txt file for review
        with open("./data/cars_sensitive_entities.txt", "a") as f:
            for ent in sensitive:
                f.write(ent + "\n")
        print(f"[+] Wrote detected entities to {base}_entities.txt")

        return

        # load sensitive terms from a file if it exists
        # 4) load YOUR sensitive terms list — this is the only set used for redaction
        sensitive = set()
        sensitive_file = "./data/sensitive_entities.txt"
        if os.path.exists(sensitive_file):
            with open(sensitive_file, "r") as f:
                sensitive.update(line.strip() for line in f if line.strip())
            # print(f"[+] Loaded sensitive terms from {sensitive_file}: {sensitive}")
        else:
            print(f"[-] No sensitive_terms file at {sensitive_file}; nothing will be redacted.")

        # 4) redact JSON + collect bleep segments
        bleep_segments = []
        for item in flat:
            txt = item["text"]
            for s in sensitive:
                if txt.lower() in s.lower().split():
                    start, end = item["start"], item["end"]
                    bleep_segments.append((start, end))
                    item["word_obj"]["Word"] = "[REDACTED]"
                    patt = r"\b" + re.escape(txt) + r"\b"
                    item["utt_obj"]["Utterance"] = re.sub(
                        patt, "[REDACTED]",
                        item["utt_obj"]["Utterance"],
                        flags=re.IGNORECASE
                    )
                    print(f"    [*] Redacted '{txt}' at {start:.3f}-{end:.3f}")
                    # log redaction along with filename
                    log_file.write(f"Redacted '{txt}' at {start:.3f}-{end:.3f}\n")
                    break

        # # 5) save redacted JSON
        # with open(out_json, "w") as f:
        #     json.dump(transcript, f, indent=2)
        # print(f"[+] Wrote redacted JSON → {out_json}")

        # 6) if no bleeps, done
        if not bleep_segments:
            print("[-] No PII spans to bleep.")
            return

        # 7) build ffmpeg filter
        # a) mute original at all spans
        vol_chain = ",".join(
            f"volume=enable='between(t,{s-0.2},{e+0.2})':volume=0"
            for s,e in bleep_segments
        )
        # b) generate & delay each beep
        beep_parts = []
        for i, (s,e) in enumerate(bleep_segments):
            dur = e - s
            ms = int(s * 1000)
            beep_parts.append(f"sine=frequency=1000:duration={dur}[b{i}];")
            beep_parts.append(f"[b{i}]adelay={ms}|{ms}[bd{i}];")
        # c) mix original+beeps
        inputs = ["[orig]"] + [f"[bd{i}]" for i in range(len(bleep_segments))]
        mix = "".join(inputs) + f"amix=inputs={len(inputs)}:duration=longest[out]"

        filter_complex = (
            f"[0]{vol_chain}[orig];"
            + "".join(beep_parts)
            + mix
        )

        cmd = [
            "ffmpeg", "-y", "-i", wav_path,
            "-filter_complex", filter_complex,
            "-map", "[out]", out_wav
        ]
        print(f"[+] Running ffmpeg to bleep {len(bleep_segments)} segments…")
        # subprocess.run(cmd, check=True)
        print(f"[+] Wrote censored audio → {out_wav}")



def main(root_dir):
    # load SpaCy once
    print("[*] Loading SpaCy en_core_web_trf…")
    nlp = spacy.load("en_core_web_trf")

    # walk recursively
    for dirpath, _, files in os.walk(root_dir):
        if os.path.basename(dirpath).lower() != "audio":
            continue
        # find all .json + matching .wav
        jsons = [f for f in files if f.lower().endswith(".json")]
        wavs  = {f for f in files if f.lower().endswith(".wav")}
        for jf in jsons:
            json_name = os.path.splitext(jf)[0]

            # strip "gemini_timestamped" if present from the end of the filename
            if json_name.endswith("_gemini_timestamped"):
                json_name = json_name[:-len("_gemini_timestamped")]
            # construct expected .wav filename
            wav_name = json_name + ".wav"
            print(f"[*] Looking for {wav_name} to match {jf} in {dirpath}")
        
            # base = os.path.splitext(jf)[0] + ".wav"
            if wav_name in wavs:
                print(f"[+] Found {jf} with matching {wav_name} in {dirpath}")
                process_pair(
                    os.path.join(dirpath, jf),
                    os.path.join(dirpath, wav_name),
                    nlp
                )
            else:
                print(f"[-] Skipping {jf}: no {wav_name} in {dirpath}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Recursively censor PII in all audio/json pairs under a root directory"
    )
    p.add_argument("root_dir", help="Top‑level folder to recurse into")
    args = p.parse_args()
    main(args.root_dir)
