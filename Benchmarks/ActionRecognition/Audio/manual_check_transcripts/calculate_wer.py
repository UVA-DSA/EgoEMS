import os
import json
import numpy as np
import jiwer
from collections import defaultdict
import inflect
import string
# Create an inflect engine
p = inflect.engine()

# Define a custom function to replace digits with words
def digits_to_words(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    converted_words = [p.number_to_words(word) if word.isdigit() else word for word in words]
    return " ".join(converted_words)

def get_wer(gt, pred):
    gt_transformed = digits_to_words(gt)
    pred_transformed = digits_to_words(pred)

    # print(gt_transformed)
    transforms = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    wer = jiwer.wer(
                gt_transformed,
                pred_transformed,
                truth_transform=transforms,
                hypothesis_transform=transforms,
            )
    return wer

def process_file(file_path):
    text = ""
    with open(file_path, "r") as f:
        conv = json.load(f)
    for i in range(len(conv)):
        text += conv[i]["Utterance"] + " "
    return text.strip()
    
def process_whisper_file(file_path):
    with open(file_path, "r") as f:
        conv = json.load(f)
    return conv["text"]

def process_gspeech_file(file_path):
    with open(file_path, "r") as f:
        conv = json.load(f)
    return conv["Full Transcript"]

def gather_gt_pred():
    path = "/scratch/zar8jw/Audio/manual_check_transcripts"
    res = {}
    for root, dirs, files in os.walk(path):
        # print(os.path.basename(root))
        if os.path.basename(root) == "wa1":
            continue
        for file in files:
            # print(root, file)
            # if "gemini" in file:
            #     tag = "pred"
            #     name = file.split("encoded_gemini")[0].strip("-").strip("_")
            #     text = process_file(os.path.join(root, file))
            
            # if "whisper_timestamp" in file:
            #     tag = "pred"
            #     name = file.split("whisper_timestamp_")[1].split("_encoded_trimmed")[0].strip("-").strip("_").strip()
            #     text = process_whisper_file(os.path.join(root, file))

            # if "whisper1" in file:
            #     tag = "pred"
            #     name = file.split("whisper1_")[1].split("_encoded_trimmed")[0].strip("-").strip("_").strip()
            #     text = process_whisper_file(os.path.join(root, file))

            if "google_speech" in file:
                tag = "pred"
                name = file.split("google_speech_")[1].split("_encoded_trimmed")[0].strip("-").strip("_").strip()
                text = process_gspeech_file(os.path.join(root, file))

            elif "human" in file:
                tag = "gt"
                name = file.split("encoded_human")[0].strip("-").strip("_")
                text = process_file(os.path.join(root, file))
            else:
                continue
            # print(name)
            if name not in res:
                res[name] = {}
            res[name][tag] = text
    return res


if __name__ == "__main__":
    res = gather_gt_pred()
    # with open("./results.json", 'w') as f:
    #     json.dump(res, f, indent=4)
    # print(res.keys())

    # for k, v in res.items():
    #     print(k, v.keys())


    for id, metadata in res.items():
        # print(id)
        cur_wer = get_wer(metadata["gt"], metadata["pred"])
        print(id, cur_wer)