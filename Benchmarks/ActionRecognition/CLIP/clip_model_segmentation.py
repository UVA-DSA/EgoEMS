import sys
import os
# sys.path.append("../MTRSAP/")
# from utils.utils import *
# from scripts.config import DefaultArgsNamespace
import torch
import torch.nn as nn
import torchvision.models as models
from functools import partial
from torch.utils.data import DataLoader

from datetime import datetime
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")
import clip
import requests
from PIL import Image
from collections import defaultdict, Counter
import json
from data_loader import WindowEgoExoEMSDataset, window_collate_fn, transform

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def none_or_int(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: {value}")

def normalize_image(input_frame):
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    input_frame = input_frame / 255.0
    input_normalized = (input_frame - clip_mean) / clip_std
    return input_normalized

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Generate Text Embedding:"):
            texts = [template.format(classname) for template in templates] #format with class
            # print(texts)
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            # print(class_embeddings.shape)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            # print(class_embedding.shape)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def clip_process(data_loader, type="train", task="segmentation", window_size="30"):
    res = {}
    pbar = tqdm(colour="blue", desc=f"{type} dataLoader", total=len(data_loader), dynamic_ncols=True)
    for i, batch in enumerate(data_loader):
        frames = batch['video']
        # print(frames.shape)
        if len(frames.shape) != 5:
            continue

        label = batch['keystep_label'][0]
        label_indexes = [str(classes.index(l)) for l in label]
        subject_id = batch['subject_id'][0][0]
        trial_id = batch['trial_id'][0][0]

        # print(subject_id, trial_id)

        if subject_id not in res:
            res[subject_id] = {}

        if trial_id not in res[subject_id]:
            res[subject_id][trial_id] = defaultdict(list)

        # num_frames = frames.shape[1]
        num_frames = len(label)
        prob = []
        indexes = []
        sequence = []
        for j in range(num_frames):
            input_frame = frames[:, j, :, :, :].to(device)

            img = Image.fromarray(input_frame[0].byte().permute(1, 2, 0).cpu().numpy())
            input_frame = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(input_frame)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ zeroshot_weights).softmax(dim=-1)
            values, index = similarity[0].topk(1)
            prob.append(str(values.cpu().numpy()[0]))
            indexes.append(str(index.cpu().numpy()[0]))
            sequence.append(classes[index])
        
        # print(len(sequence), len(label))
        assert len(sequence) == len(label), f"{i} in {type}: sequence:{len(sequence)}, label:{len(label)}"
    #     pred_key_step = majority_voter(sequence_1s)

        res[subject_id][trial_id]["pred"].extend(sequence)
        res[subject_id][trial_id]["pred index"].extend(indexes)
        res[subject_id][trial_id]["label"].extend(label)
        res[subject_id][trial_id]["label index"].extend(label_indexes)
        pbar.update(1)

        if not os.path.exists("./results"):
            os.makedirs("./results")
        
        with open(f"./results/{type}_{task}_{window_size}.json", 'w') as f:
            json.dump(res, f, indent=4)
            

    pbar.close()
    return res


def dataloader(args):
    print("start data loading")
    collate_fn_with_args = partial(window_collate_fn, frames_per_clip=args.observation_window)

    alldata_dataset = WindowEgoExoEMSDataset(annotation_file=args.train_annotation_path,
                                     data_base_path='',
                                    fps=args.fps, 
                                    frames_per_clip=args.observation_window, 
                                    transform=transform, 
                                    data_types=args.modality)
    alldata_loader = DataLoader(alldata_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_args)
    print("finished dataloading")
    return alldata_loader


if __name__ == "__main__":

    keysteps = {
    "approach_patient": "Approach the patient",
    "check_responsiveness": "Check for responsiveness",
    "check_pulse": "Check patient's pulse",
    "check_breathing": "Check if patient is breathing",
    "chest_compressions": "Perform chest compressions",
    "request_aed": "Request an AED",
    "request_assistance": "Request additional assistance",
    "turn_on_aed": "Turn on the AED",
    "attach_defib_pads": "Attach defibrillator pads",
    "clear_for_analysis": "Clear for analysis",
    "clear_for_shock": "Clear for shock",
    "administer_shock_aed": "Administer shock using AED",
    "open_airway": "Open patient's airway",
    "place_bvm": "Place bag valve mask (BVM)",
    "ventilate_patient": "Ventilate patient",
    "no_action": "No action",
    "assess_patient": "Assess the patient",
    "explain_procedure": "Explain the ECG procedure to the patient",
    "shave_patient": "Shave/Cleanse the patient for ECG",
    "place_left_arm_lead": "Place the lead on left arm for ECG",
    "place_right_arm_lead": "Place the lead on right arm for ECG",
    "place_left_leg_lead": "Place the lead on left leg for ECG",
    "place_right_leg_lead": "Place the lead on right leg for ECG",
    "place_v1_lead": "Place the V1 lead on the patient",
    "place_v2_lead": "Place the V2 lead on the patient",
    "place_v3_lead": "Place the V3 lead on the patient",
    "place_v4_lead": "Place the V4 lead on the patient",
    "place_v5_lead": "Place the V5 lead on the patient",
    "place_v6_lead": "Place the V6 lead on the patient",
    "ask_patient_age_sex": "Ask the age and or sex of the patient",
    "request_patient_to_not_move": "Request the patient to not move",
    "turn_on_ecg": "Turn on the ECG machine",
    "connect_leads_to_ecg": "Verify all ECG leads are properly connected",
    "obtain_ecg_recording": "Obtain the ECG recording",
    "interpret_and_report": "Interpret the ECG and report findings"
    }

    # add more templates from the paper 
    # https://arxiv.org/pdf/2403.16182#page=15.08
    prompt_tempate = [
        "A Photo of a person {}.",
        "First Responder is {} in the picture.",
        "The image shows a first responder {}.",
        "The Emergency Medical Service task being performed is {}.",
        "The first responder is currently {}.",
        "A person is seen {}.",
        "Can you recognize the action of {}?",
        "Playing action of {}.",
        "Video classification of {}.",
        "A video of {}.",
        "Look, the human is {}.",
        "{}, an action."
    ]

    # prompt_tempate =  ["A Photo of a person {}"]
    classes = list(keysteps.keys())
    keystep_descriptions = list(keysteps.values())
    zeroshot_weights = zeroshot_classifier(keystep_descriptions, prompt_tempate)

    parser = argparse.ArgumentParser(description="Training script for recognition")
    parser.add_argument('--data_split', type=str, default="test")
    parser.add_argument('--train_annotation_path', type=str, default="/scratch/zar8jw/EgoExoEMS/Annotations/splits/trials/test_split_segmentation.json")
    parser.add_argument('--task', type=str, default="segmentation")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--observation_window', type=none_or_int, default=120)
    parser.add_argument('--modality', type=list, default=["video"])
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    all_loader = dataloader(args)

    all_res = clip_process(all_loader, type=args.data_split, task=args.task, window_size=args.observation_window)

    with open(f"./results/{args.data_split}_{args.task}_{args.observation_window}.json", 'w') as f:
        json.dump(all_res, f, indent=4)
    print("finished")



