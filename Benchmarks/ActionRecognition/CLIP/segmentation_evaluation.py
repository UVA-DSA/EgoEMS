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


# Function to calculate IoU (Intersection over Union)
def calculate_iou(pred_interval, gt_intervals):
    intersection = np.minimum(pred_interval[1],gt_intervals[:,1]) - np.maximum(pred_interval[0],gt_intervals[:,0])
    union = np.maximum(pred_interval[1],gt_intervals[:,1]) - np.minimum(pred_interval[0],gt_intervals[:,0])
    return (intersection / union)

def calculate_f1_at_k(ground_truth, prediction, k, n_classes=3):
    # Break the ground truth and prediction into segments with start and end indices
    gt_intervals = []
    pred_intervals = []
    
    gt_labels = []
    pred_labels = []
    # Find start and end of each ground truth segment
    start_idx = 1
    for i in range(1, len(ground_truth)):
        if ground_truth[i] != ground_truth[i - 1]:
            gt_intervals.append((start_idx, i ))
            gt_labels.append(ground_truth[i-1])
            start_idx = i+1
    # Append the last ground truth segment
    gt_intervals.append((start_idx, len(ground_truth)))
    gt_labels.append(ground_truth[-1])

    # Find start and end of each prediction segment
    start_idx = 1
    for i in range(1, len(prediction)):
        if prediction[i] != prediction[i - 1]:
            pred_intervals.append((start_idx, i))
            pred_labels.append(prediction[i-1])
            start_idx = i+1
    # Append the last prediction segment
    pred_intervals.append((start_idx, len(prediction)))
    pred_labels.append(prediction[-1])


    gt_intervals = np.array(gt_intervals)
    pred_intervals = np.array(pred_intervals)

        # We keep track of the per-class TPs, and FPs.
    # In the end we just sum over them though.
    TP = np.zeros(n_classes+1, float)
    FP = np.zeros(n_classes+1, float)


    n_pred = len(pred_intervals)
    n_true = len(gt_intervals)

    true_used = np.zeros(n_true, float)

    for i in range(n_pred):
        current_pred_interval = pred_intervals[i]
        current_pred_label = pred_labels[i]

        IoU = calculate_iou(current_pred_interval, gt_intervals)*(np.array(gt_labels) == current_pred_label)

        idx = np.argmax(IoU)

        # If the IoU is high enough and the true segment isn't already used
        # Then it is a true positive. Otherwise is it a false positive.
        if IoU[idx] >= k and not true_used[idx]:
            TP[pred_labels[i]] += 1
            true_used[idx] = 1
        else:
            FP[pred_labels[i]] += 1


    TP = TP.sum()
    FP = FP.sum()
    # False negatives are any unused true segment (i.e. "miss")
    FN = n_true - true_used.sum()
    
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    F1 = 2 * (precision*recall) / (precision+recall)

    # If the prec+recall=0, it is a NaN. Set these to 0.
    F1 = np.nan_to_num(F1)

    return F1, precision, recall, TP, FP, FN


def calculate_acc_editscore(gt, pred):
    acc = accuracy_score(gt, pred)
    edit_dist = editdistance.eval(pred, gt)
    edit_score = 1 - (edit_dist / max(len(pred), len(gt)))
    return acc, edit_dist, edit_score

if __name__ == "__main__":

    # gather_pred_gt()
    thresh = 0.5

    classes = list(keysteps.keys())

    with open("./results/test_segmentation_120.json", "r") as f:
        res = json.load(f)

    metrics = {}
    for subject, trials in res.items():
        for trial, metadata in trials.items():
            pred = [int(each) for each in metadata["pred index"]]
            gt = [int(each) for each in metadata["label index"]]
  
            F1, precision, recall, TP, FP, FN = calculate_f1_at_k(gt, pred, k=thresh, n_classes=len(keysteps))
            acc, edit_dist, edit_score = calculate_acc_editscore(gt, pred)
            metrics[subject + '_' + trial] = {
                f"f1@{thresh}": F1,
                f"p@{thresh}": precision,
                f"r@{thresh}": recall,
                "acc": acc,
                "edit distance": edit_dist,
                "edit score": edit_score
            }
    

    p, r, f1, acc, ed, es = 0, 0, 0, 0, 0, 0
    cnt = 0
    for k, v in metrics.items():
        cnt += 1
        p += v[f"f1@{thresh}"]
        r += v[f"p@{thresh}"]
        f1 += v[f"r@{thresh}"]
        acc += v['acc']
        ed += v['edit distance']
        es += v['edit score']
    
    performance = {
        "p": p / cnt,
        "r": r / cnt,
        "f1": f1 / cnt,
        "acc": acc / cnt,
        "edit distance": ed / cnt,
        "edit score": es / cnt
    }
    print(f"IOU={thresh}")
    print(performance)
        

    


    
    