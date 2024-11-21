import numpy as np
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error
import editdistance
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

def calculate_iou(gt_interval, pred_interval):
    start_gt, end_gt = gt_interval
    start_pred, end_pred = pred_interval
    
    # Calculate intersection
    inter_start = max(start_gt, start_pred)
    inter_end = min(end_gt, end_pred)
    if inter_start >= inter_end:  # No overlap
        return 0.0
    intersection = inter_end - inter_start
    
    # Calculate union
    union = (end_gt - start_gt) + (end_pred - start_pred) - intersection
    return intersection / union

# Function to match events based on labels and IoU
def match_events(pred, gt, iou_threshold=0.5):
    true_positives = []
    false_positives = []
    false_negatives = []
    
    used_gt_indices = set()
    
    # Loop over predicted events and try to match with GT
    for p_label, p_start, p_end in pred:
        matched = False
        for idx, (gt_label, gt_start, gt_end) in enumerate(gt):
            if idx in used_gt_indices:
                continue  # Skip already matched GT events
            
            # Match label and IoU > threshold
            if p_label == gt_label and calculate_iou([gt_start, gt_end], [p_start, p_end]) > iou_threshold:
                true_positives.append((p_label, p_start, p_end))
                used_gt_indices.add(idx)
                matched = True
                break
        
        if not matched:
            false_positives.append((p_label, p_start, p_end))
    
    # Any unmatched GT events are false negatives
    for idx, (gt_label, gt_start, gt_end) in enumerate(gt):
        if idx not in used_gt_indices:
            false_negatives.append((gt_label, gt_start, gt_end))
    
    return true_positives, false_positives, false_negatives


# Function to calculate MAE for matched events
def calculate_mae(true_positives, gt):
    gt_dict = {label: (start, end) for label, start, end in gt}
    
    start_errors = []
    end_errors = []
    
    # Loop over matched events to calculate time errors
    for label, pred_start, pred_end in true_positives:
        gt_start, gt_end = gt_dict[label]
        start_errors.append(abs(pred_start - gt_start))
        end_errors.append(abs(pred_end - gt_end))
    
    # Calculate mean absolute error for start and end times
    mae_start = mean_absolute_error([gt_dict[label][0] for label, _, _ in true_positives], [pred_start for _, pred_start, _ in true_positives])
    mae_end = mean_absolute_error([gt_dict[label][1] for label, _, _ in true_positives], [pred_end for _, _, pred_end in true_positives])
    
    return mae_start, mae_end


# Function to calculate accuracy, edit distance, edit score (Levenshtein distance)
def calculate_edit_score(pred, gt):
    classes = list(keysteps.keys())
    pred_labels = [classes.index(label) for label, _, _ in pred]
    gt_labels = [classes.index(label) for label, _, _ in gt]

    edit_dist = editdistance.eval(pred_labels, gt_labels)
    edit_score = 1 - (edit_dist / max(len(pred_labels), len(gt_labels)))
    return edit_dist, edit_score


def calculate_all_metrics_timelevel(pred, gt, iou_threshold=0.5):
    # Event matching
    true_positives, false_positives, false_negatives = match_events(pred, gt, iou_threshold)
    
    # Precision, Recall, F1
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # # MAE (for start and end times)
    # mae_start, mae_end = calculate_mae(true_positives, gt)
    
    # Edit Score
    edit_dist, edit_score = calculate_edit_score(pred, gt)
    
    return {
        f"p@{iou_threshold}": precision,
        f"r@{iou_threshold}": recall,
        f"f1@{iou_threshold}": f1,
        # "acc": acc,
        "edit distance": edit_dist,
        "edit score": edit_score
    }

def calculate_all_metrics_framelevel(pred, gt, iou_threshold=0.5):
    # Event matching
    true_positives, false_positives, false_negatives = match_events(pred, gt, iou_threshold)
    
    # Precision, Recall, F1
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    max_end_t = max(pred[-1][-1], gt[-1][-1])
    pred_frame = time2frame(pred, max_end_t, fps=30)
    gt_frame = time2frame(gt, max_end_t, fps=30)
    print(len(pred_frame), len(gt_frame))
    assert len(pred_frame) == len(gt_frame)
    acc = accuracy_score(gt_frame, pred_frame)

    # print(pred_frame)

    edit_dist = editdistance.eval(pred_frame, gt_frame)
    edit_score = 1 - (edit_dist / max(len(pred_frame), len(gt_frame)))
    return {
        f"p@{iou_threshold}": precision,
        f"r@{iou_threshold}": recall,
        f"f1@{iou_threshold}": f1,
        "acc": acc,
        "edit distance": edit_dist,
        "edit score": edit_score
    }


def gather_pred_gt():
    """
    collect corresponding groundtruth and predictions
    save as all_results.json
    """
    if not os.path.exists('./results'):
        os.makedirs("./results")

    with open("/scratch/zar8jw/EgoExoEMS/Annotations/splits/trials/test_split_segmentation.json", 'r') as f:
        gt = json.load(f)
    
    n = len(gt["subjects"])

    all_dict = {}

    for i in range(n):
        subject_id = gt["subjects"][i]["subject_id"]
        num_trials = len(gt["subjects"][i]["trials"])
        for j in range(num_trials):
            trial_id = gt["subjects"][i]["trials"][j]["trial_id"]
            raw_sequence = gt["subjects"][i]["trials"][j]["keysteps"]

            # for each in gt["subjects"][i]["trials"][j]["streams"]["audio"]:
            #     if "keystep_timestamp" in each["file_path"]:
            #         file_path = each["file_path"]

            # if "keystep_timestamp" not in file_path:
            #     print(file_path)
            #     raise Exception("wrong prediction file path")
            
            # with open(file_path, "r") as f:
            #     pred = json.load(f)

            if f"{subject_id}_{trial_id}_pred.json" not in os.listdir("./logs/segmentation/test/intermediate_result"):
                continue

            with open(f"./logs/segmentation/test/intermediate_result/{subject_id}_{trial_id}_pred.json", 'r') as f:
                pred = json.load(f)


            trial_seq = []
            for k in range(len(raw_sequence)):
                label = raw_sequence[k]["label"]
                start_t = raw_sequence[k]["start_t"]
                end_t = raw_sequence[k]["end_t"]
                cur_keystep = [label, start_t, end_t]
                trial_seq.append(cur_keystep)

            if trial_seq:
                
                cur_dict = {
                    "Prediction": pred,
                    "Groundtruth": trial_seq
                }
                all_dict[f"{subject_id}_{trial_id}"] = cur_dict
        
    with open("./results/test_segmentation.json", 'w') as f:
        json.dump(all_dict, f, indent=4)

def process_seq(seq):

    def is_list_of_lists(data):
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, list):
                return False
        return True


    processed = []
    n = len(seq)
    for i in range(n):
        if is_list_of_lists(seq[i]):
            for sub_seq in seq[i]:
                if sub_seq[0] == 'none' or sub_seq[0] == 'no action':
                    continue
                processed.append(sub_seq)
        else:
            processed.append(seq[i])
    return processed


def time2frame(seq, max_end, fps=30):
    classes = list(keysteps.keys())
    frame_label = [-1] * round(max_end * fps)
    # print(len(frame_label))
    for i, (event, start_t, end_t) in enumerate(seq):
        start_t = round(start_t, 2)
        end_t = round(end_t, 2)
        if i == 0:
            start_t = min(0, start_t)
        
        if i == len(seq)-1:
            end_t = max(end_t, max_end)
        
        start_f = round(start_t * fps)
        end_f = round(end_t * fps)
        idx = classes.index(event)
        # print(idx, start_f, end_f)
        
        frame_label[start_f:end_f] = [idx] * (end_f-start_f)

    return frame_label



if __name__ == "__main__":

    gather_pred_gt()

    with open("./results/test_segmentation.json", "r") as f:
        res = json.load(f)
    
    metrics = {}
    iou_threshold = 0.5
    for k, v in tqdm(res.items()):
        pred = process_seq(v["Prediction"])
        gt = process_seq(v["Groundtruth"])
        print(k)
        trial_metrics = calculate_all_metrics_framelevel(pred, gt, iou_threshold)
        metrics[k] = trial_metrics


    p, r, f1, acc, ed, es = 0, 0, 0, 0, 0, 0
    cnt = 0
    for k, v in metrics.items():
        cnt += 1
        p += v[f"f1@{iou_threshold}"]
        r += v[f"p@{iou_threshold}"]
        f1 += v[f"r@{iou_threshold}"]
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
    print(f"IOU={iou_threshold}")
    print(performance)

    
    