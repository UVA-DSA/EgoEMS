import torch


RECORD_RESULTS = True

tcn_model_params = {
    "encoder_params": { #some of these gets updated during runtime based on the feature dimension of the given data
        "in_channels": 14,
        "kernel_size": 1,
        "out_channels": 64,
    },
    "decoder_params": {
        "in_channels": 64,
        "kernel_size": 31,
        "out_channels": 64
    }
}


transformer_params = {
    "d_model": 1024,
    "nhead": 4,
    "num_layers": 4,
    "hidden_dim": 128,
    "layer_dim": 4,
    "dropout": 0.1,
    "input_dim": 64,
    "output_dim": 16,
    "batch_first": True

}

learning_params = {
    # "lr": 8.906324028628413e-5,
    "lr": 1e-05,
    "epochs": 30,
    "weight_decay": 1e-5,
    "patience": 3,
    "lr_drop": 15,
    "best_chkpoint": "64366030/val_best_model.pt"
}

dataloader_params = {
    
    "batch_size": 1,
    "observation_window": 120,
    "fold": 1,
    "fps": 30,
    "train_annotation_path": '/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Annotations/splits/keysteps/train_split.json',
    "val_annotation_path": '/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Annotations/splits/keysteps/val_split.json',
    "test_annotation_path": '/scratch/cjh9fw/Rivanna/2024/repos/EgoExoEMS/Annotations/splits/keysteps/test_split.json',
    # Old dataset class
    'base_path': '/scratch/cjh9fw/Rivanna/2024/datasets/EMS_Datasets/Organized/EMS_Interventions/annotations/',
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