import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import editdistance

def calculate_iou(pred_interval, gt_intervals):

    intersection = np.minimum(pred_interval[1],gt_intervals[:,1]) - np.maximum(pred_interval[0],gt_intervals[:,0])
    union = np.maximum(pred_interval[1],gt_intervals[:,1]) - np.minimum(pred_interval[0],gt_intervals[:,0])
    return (intersection / union)


"""
Calculate the F1 score for a segmentation task.

Calculate the F1 score at a given threshold k for action recognition.
This function computes the F1 score, precision, recall, true positives (TP), 
false positives (FP), and false negatives (FN) based on the intersection over 
union (IoU) of predicted and ground truth segments.
Parameters:
-----------
ground_truth : list or array-like [1,1,2,2,3,3,...]
    The ground truth labels for each time step.
prediction : list or array-like  [1,1,2,3,3,3,...]
    The predicted labels for each time step.
k : float
    The IoU threshold to consider a prediction as a true positive.
n_classes : int, optional
    The number of classes (default is 3).
Returns:
--------
F1 : float
    The F1 score at the given threshold k.
precision : float
    The precision at the given threshold k.
recall : float
    The recall at the given threshold k.
TP : float
    The number of true positives.
FP : float
    The number of false positives.
FN : float
    The number of false negatives.
"""

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

    print("n_pred: ", n_pred)
    print("n_true: ", n_true)
    true_used = np.zeros(n_true, float)

    for i in range(n_pred):
        current_pred_interval = pred_intervals[i]
        current_pred_label = pred_labels[i]

        IoU = calculate_iou(current_pred_interval, gt_intervals)*(np.array(gt_labels) == current_pred_label)

        idx = np.argmax(IoU)

        # If the IoU is high enough and the true segment isn't already used
        # Then it is a true positive. Otherwise is it a false positive.
        print("i", i)
        print("pred_labels[i]", pred_labels[i])
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


"""
Calculate the F1 score for a classification task.
Parameters:
-----------
ground_truth : list or array-like
    The ground truth labels for each sample.
prediction : list or array-like
    The predicted labels for each sample.
Returns:
--------
F1 : float
    The F1 score for the classification task.
precision : float
    The precision for the classification task.
recall : float
    The recall for the classification task.
"""
def calculate_f1_classification(ground_truth, prediction):

    F1 = f1_score(ground_truth, prediction, average='macro')
    precision = precision_score(ground_truth, prediction, average='macro')
    recall = recall_score(ground_truth, prediction, average='macro')
    return F1, precision, recall






# Function to calculate final average class-wise accuracy
def calculate_final_classwise_accuracy(all_ground_truth, all_predicted, actions):
    correct_predictions_per_class = {action: 0 for action in actions}
    total_samples_per_class = {action: 0 for action in actions}

    for ground_truth, predicted in zip(all_ground_truth, all_predicted):
        for action in actions:
            true_binary = np.array(ground_truth) == action
            pred_binary = np.array(predicted) == action
            
            correct_predictions = np.sum(true_binary & pred_binary)
            total_samples = np.sum(true_binary)

            correct_predictions_per_class[action] += correct_predictions
            total_samples_per_class[action] += total_samples

    # Calculate accuracy for each class
    classwise_accuracy = {}
    for action in actions:
        total_samples = total_samples_per_class[action]
        if total_samples > 0:
            classwise_accuracy[action] = np.round(correct_predictions_per_class[action] / total_samples, 4)
        else:
            classwise_accuracy[action] = 0.0  # If no samples for this class

    return classwise_accuracy



# Function to calculate edit score
def calculate_edit_score(ground_truth, predicted, edit_distance):
    max_len = max(len(ground_truth), len(predicted))
    if max_len == 0:  # To avoid division by zero
        return 1.0
    edit_score = 1 - (edit_distance / max_len)
    return np.round(edit_score, 4)

# Function to calculate Levenshtein distance (edit distance)
def calculate_edit_distance(ground_truth, predicted):
    edit_distance = editdistance.eval(ground_truth, predicted)
    return edit_distance
