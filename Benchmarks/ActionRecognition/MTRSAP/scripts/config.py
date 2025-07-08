import torch


RECORD_RESULTS = True

tcn_model_params = {
    "encoder_params": { #some of these gets updated during runtime based on the feature dimension of the given data
        "in_channels": 1024,
        "kernel_size": 45,
        "out_channels": 256,
    },
    "decoder_params": {
        "in_channels": 60,
        "kernel_size": 31,
        "out_channels": 60
    }
}


transformer_params = {
    "d_model": 256,
    "nhead": 4,
    "num_layers": 2,
    "hidden_dim": 128,
    "layer_dim": 4,
    "dropout": 0.1,
    "input_dim": 1024,
    "output_dim": 16,
    "batch_first": True,
    # Parameters for audio feature extraction
    'sample_rate' : 48000,  # Adjust based on your dataset
    'n_mels' : 64,  # Number of Mel filter banks
    'hop_length' : 512,
    'n_fft' : 1024,
    # resnet feature dim
    'resnet_dim' : 2048


}

learning_params = {
    # "lr": 8.906324028628413e-5,
    "lr": 1e-05,
    "epochs": 30,
    "weight_decay": 1e-5,
    "patience": 3,
    "lr_drop": 20,
    "best_chkpoint": "/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Benchmarks/ActionRecognition/MTRSAP/checkpoints/job_755732_task_classification/val_best_model.pt"
}

dataloader_params = {
    
    "task": "segmentation", # "segmentation" or "classification"
    "batch_size": 1,
    "observation_window": 150,  # 5 seconds at 30 fps segmentation :::: classification None
    "fold": 1,
    "fps": 29.97,
    # update task specific parameters (Experimenting segmentation with classification annotations)
    # "train_annotation_path": '/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Annotations/splits/trials/aaai26_train_split_classification.json',
    # "val_annotation_path": '/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Annotations/splits/trials/aaai26_val_split_classification.json',
    # "test_annotation_path": '/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Annotations/splits/trials/aaai26_test_split_classification.json',
    "train_annotation_path": '/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Annotations/splits/trials/aaai26_train_split_segmentation.json',
    "val_annotation_path": '/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Annotations/splits/trials/aaai26_val_split_segmentation.json',
    "test_annotation_path": '/home/cjh9fw/Desktop/2024/repos/EgoExoEMS/Annotations/splits/trials/aaai26_test_split_segmentation.json',
    # Old dataset class
    'base_path': '/home/cjh9fw/Desktop/2024/datasets/EMS_Datasets/Organized/EMS_Interventions/annotations/',
    'modality': [ 'resnet_ego'],
    'keysteps' : {
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
                    "interpret_and_report": "Interpret the ECG and report findings",
                    "transport": "Transport the patient to the hospital",
                    "check_grip_strength": "Check grip strength",
                    "check_symptom_duration": "Check symptom duration",
                    "review_medications": "Review medications",
                    "inquire_medication_anticoagulants": "Inquire about anticoagulant medications",
                    "inquire_hpi_and_pmh": "Inquire about HPI and PMH",
                    "inquire_substance_use": "Inquire about substance use",
                    "notify_hospital_of_stroke_alert": "Notify hospital of stroke alert",
                    "check_blood_pressure": "Check blood pressure",
                    "check_heart_rate": "Check heart rate",
                    "check_oxygen_saturation": "Check oxygen saturation",
                    "check_respiratory_rate": "Check respiratory rate",
                    "face_droop_check": "Check for facial droop",
                    "arm_drift_check": "Check for arm drift",
                    "speech_abnormality_check": "Check for speech abnormalities",
                    "assess_balance_and_coordination": "Assess balance and coordination",
                    "document_lkw_time": "Document last known well time",
                    "check_vision_deficits": "Check for vision deficits",
                    "evaluate_aphasia": "Evaluate for aphasia",
                    "assess_neglect_signs": "Assess for neglect signs",
                    "prepare_glucometer_and_strip": "Prepare glucometer and test strip",
                    "read_and_record_glucose_level": "Read and record glucose level",
                    "suction_airway": "Suction airway",
                    "inset_NPA": "Insert NPA",
                    "load_patient_to_stretcher": "Load patient to stretcher",
                    "secure_patient_on_stretcher": "Secure patient on stretcher",
                    "handoff_patient_to_hospital": "Handoff patient to hospital staff",
                    "check_perrl": "Check PERRL",
                    "check_skin_condition": "Check skin condition",
                    "check_a&o": "Check A&O",
                    "notify_hospital": "Notify hospital",
                    "document_hpi_and_pmh": "Document HPI and PMH"
                },
            "train_class_stats": {},
            "val_class_stats": {}

}


class DefaultArgsNamespace:
    def __init__(self):
        
        self.record_results = RECORD_RESULTS

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model parameters
        self.tcn_model_params = tcn_model_params
        self.transformer_params = transformer_params

        # Learning parameters
        self.learning_params = learning_params

        # DataLoader parameters
        self.dataloader_params = dataloader_params