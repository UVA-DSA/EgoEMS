import numpy as np
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import editdistance
from sklearn.metrics import mean_absolute_error
from nltk.metrics.distance import edit_distance
from tqdm import tqdm

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


def cal_metrics(pred, gt):
    micro_p = precision_score(gt, pred, average='micro', zero_division=0)
    micro_r = recall_score(gt, pred, average='micro', zero_division=0)
    micro_f1 = f1_score(gt, pred, average='micro', zero_division=0)
    
    macro_p = precision_score(gt, pred, average='macro')
    macro_r = recall_score(gt, pred, average='macro')
    macro_f1 = f1_score(gt, pred, average='macro')

    acc = accuracy_score(gt, pred)

    edit_dist = editdistance.eval(pred, gt)

    # Calculate edit score
    edit_score = 1 - (edit_dist / max(len(pred), len(gt)))
    return {
        "micro": [micro_p, micro_r, micro_f1],
        "macro": [macro_p, macro_r, macro_f1],
        "acc": acc,
        "edit distance": edit_dist,
        "edit score": edit_score
    }



if __name__ == "__main__":

    # gather_pred_gt()

    classes = list(keysteps.keys())

    with open("./results/test_classification.json", "r") as f:
        res = json.load(f)
    

    metrics = {}
    for subject, trials in res.items():
        for trial, metadata in trials.items():
            all_preds = []
            all_gt = []
            pred = metadata["pred"]
            gt = metadata["label"]
            for p in pred:
                if type(p) == list:
                    p = p[0]
                first_p = p.split(',')[0].strip()
                if first_p in classes:
                    all_preds.append(classes.index(first_p))
                else:
                    all_preds.append(classes.index("no_action"))
            
            for g in gt:
                all_gt.append(classes.index(g))
            
            cur_metric = cal_metrics(all_preds, all_gt)
            metrics[subject + '_' + trial] = cur_metric

    p, r, f1, acc, ed, es = 0, 0, 0, 0, 0, 0
    cnt = 0
    for k, v in metrics.items():
        cnt += 1
        p += v['macro'][0]
        r += v['macro'][1]
        f1 += v['macro'][2]
        acc += v['acc']
        ed += v['edit distance']
        es += v['edit score']
    
    performance = {
        "p": p / cnt,
        "r": r / cnt,
        "f1": f1 / cnt,
        "acc": acc / cnt,
        "ed": ed / cnt,
        "es": es / cnt
    }
    print(performance)
        

    


    
    