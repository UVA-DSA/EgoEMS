import sys
import os
# sys.path.append("../MTRSAP/")
# from utils.utils import *
# from scripts.config import DefaultArgsNamespace
import torch
import torch.nn as nn
import torchvision.models as models
# from datautils.ems import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
# import wandb
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
from data_loader import EgoExoEMSDataset, collate_fn, transform

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


def majority_voter(keystep_seq):
    freq = Counter(keystep_seq)
    return max(freq, key=freq.get)


def clip_process(data_loader, type="train", task="classification"):
    log = {}
    res = {}
    pbar = tqdm(colour="blue", desc=f"{type} dataLoader", total=len(data_loader), dynamic_ncols=True)
    for i, batch in enumerate(data_loader):
        frames = batch['frames']
        label = batch['keystep_label']
        subject_id = batch['subject_id'][0]
        trial_id = batch['trial_id'][0]
        start_t = batch['start_t'][0]
        end_t = batch['end_t'][0]
        if subject_id not in res:
            res[subject_id] = {}
        if subject_id not in log:
            log[subject_id] = {}

        if trial_id not in log[subject_id]:
            log[subject_id][trial_id] = defaultdict(list)
        if trial_id not in res[subject_id]:
            res[subject_id][trial_id] = defaultdict(list)

        sequence_1s = []
        prob_1s = []
        num_frames = frames.shape[1]
        for j in range(num_frames):
            input_frame = frames[:, j, :, :, :].to(device)

            img = Image.fromarray(input_frame[0].byte().permute(1, 2, 0).cpu().numpy())
            input_frame = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(input_frame)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ zeroshot_weights).softmax(dim=-1)
            values, index = similarity[0].topk(1)
            prob_1s.append(str(values.cpu().numpy()[0]))
            sequence_1s.append(classes[index])

        pred_key_step = majority_voter(sequence_1s)

        res[subject_id][trial_id]["pred"].append(pred_key_step)
        n = len(label)
        
        res[subject_id][trial_id]["label"].append(label[0])
        # print(start_t.cpu().numpy())
        # print(end_t.cpu().numpy())
        res[subject_id][trial_id]["time"].append([str(start_t.cpu().numpy()), str(end_t.cpu().numpy())])

        log[subject_id][trial_id]["seq 1s"].append(sequence_1s)
        log[subject_id][trial_id]["prob 1s"].append(prob_1s)
        pbar.update(1)

        if i % 10 == 0:
            if not os.path.exists("./results"):
                os.makedirs("./results")
            
            if not os.path.exists("./logs"):
                os.makedirs("./logs")

            with open(f"./results/{type}_{task}.json", 'w') as f:
                json.dump(res, f, indent=4)
            
            with open(f"./logs/{type}_{task}.json", 'w') as f:
                json.dump(log, f, indent=4)

    pbar.close()
    return res, log


def dataloader(args):
    alldata_dataset = EgoExoEMSDataset(annotation_file=args.train_annotation_path,
                                     data_base_path='',
                                    fps=args.fps, 
                                    frames_per_clip=args.observation_window, 
                                    transform=transform, 
                                    data_types=args.modality)
    alldata_loader = DataLoader(alldata_dataset, batch_size=args.batch_size, shuffle=False)
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
    parser.add_argument('--train_annotation_path', type=str, default="/scratch/zar8jw/EgoExoEMS/Annotations/splits/trials/test_split_classification.json")
    parser.add_argument('--task', type=str, default="classification")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--observation_window', type=none_or_int, default=None)
    parser.add_argument('--modality', type=list, default=["video"])
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    all_loader = dataloader(args)

    all_res, all_log = clip_process(all_loader, type=args.data_split, task=args.task)
    with open(f"./results/{args.data_split}_{args.task}.json", 'w') as f:
        json.dump(all_res, f, indent=4)
    with open(f"./logs/{args.data_split}_{args.task}.json", 'w') as f:
        json.dump(all_log, f, indent=4)
    
    print("finished")

    # val_res, val_log = clip_process(val_loader, type="val")
    # with open("./results/val.json", 'w') as f:
    #     json.dump(val_res, f, indent=4)
    # with open("./logs/val.json", 'w') as f:
    #     json.dump(val_log, f, indent=4)


    # test_res, test_log = clip_process(test_loader, type="test")
    # with open("./results/test.json", 'w') as f:
    #     json.dump(test_res, f, indent=4)
    # with open("./logs/test.json", 'w') as f:
    #     json.dump(test_log, f, indent=4)


