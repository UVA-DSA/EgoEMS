import numpy as np
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error
from nltk.metrics.distance import edit_distance
from tqdm import tqdm

# Helper function to calculate IoU between two intervals
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


# Function to calculate Edit Score (Levenshtein distance)
def calculate_edit_score(pred, gt):
    pred_labels = [label for label, _, _ in pred]
    gt_labels = [label for label, _, _ in gt]
    
    # Compute normalized edit distance (Levenshtein distance)
    edit_dist = edit_distance(pred_labels, gt_labels)
    max_len = max(len(pred_labels), len(gt_labels))
    edit_score = 1 - (edit_dist / max_len) if max_len > 0 else 1.0  # Normalize the distance to get the score
    
    return edit_score


def calculate_all_metrics(pred, gt, iou_threshold=0.5):
    # Event matching
    true_positives, false_positives, false_negatives = match_events(pred, gt, iou_threshold)
    
    # Precision, Recall, F1
    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # MAE (for start and end times)
    mae_start, mae_end = calculate_mae(true_positives, gt)
    
    # Edit Score
    edit_score = calculate_edit_score(pred, gt)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mae_start": mae_start,
        "mae_end": mae_end,
        "edit_score": edit_score
    }



def gather_pred_gt():
    """
    collect corresponding groundtruth and predictions
    save as all_results.json
    """
    if not os.path.exists('./results'):
        os.makedirs("./results")

    with open("./results/main_annotation.json", 'r') as f:
        gt = json.load(f)
    
    n = len(gt["subjects"])

    all_dict = {}

    for i in range(n):
        subject_id = gt["subjects"][i]["subject_id"]
        num_trials = len(gt["subjects"][i]["trials"])
        for j in range(num_trials):
            trial_id = gt["subjects"][i]["trials"][j]["trial_id"]
            raw_sequence = gt["subjects"][i]["trials"][j]["keysteps"]

            for each in gt["subjects"][i]["trials"][j]["streams"]["audio"]:
                if "keystep_timestamp" in each["file_path"]:
                    file_path = each["file_path"]

            if "keystep_timestamp" not in file_path:
                print(file_path)
                raise Exception("wrong prediction file path")
            
            with open(file_path, "r") as f:
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
                all_dict[file_path.split("/audio")[0].strip()] = cur_dict
        
    with open("./results/all_results.json", 'w') as f:
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


def cal_mean_performance(performance):
    p, r, f1, st_mae, ed_mae, edit_score = 0, 0, 0, 0, 0, 0
    for k, v in performance.items():
        p += v["precision"]
        r += v["recall"]
        f1 += v["f1_score"]
        st_mae += v["mae_start"]
        ed_mae += v["mae_end"]
        edit_score += v["edit_score"]
    n = len(performance)

    return {
        "precision": p / n,
        "recall": r / n,
        "f1-score": f1 / n,
        "mae_start": st_mae / n,
        "mae_end": ed_mae / n,
        "edit_score": edit_score / n,
    }



if __name__ == "__main__":
    with open("./results/all_results.json", "r") as f:
        res = json.load(f)
    
    evaluation_res = {}
    iou_threshold = 0
    for k, v in tqdm(res.items()):
        pred = process_seq(v["Prediction"])
        gt = process_seq(v["Groundtruth"])
        metrics = calculate_all_metrics(pred, gt, iou_threshold)
        evaluation_res[k] = metrics
    
    with open(f"./results/eval_results_IOU{iou_threshold}.json", 'w') as f:
        json.dump(evaluation_res, f, indent=4)
    
    mean_res = cal_mean_performance(evaluation_res)
    print(f"setting IOU as {iou_threshold}, the performance is")
    print(mean_res)
    

    
    