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
    "d_model": 256,
    "nhead": 4,
    "num_layers": 4,
    "hidden_dim": 128,
    "layer_dim": 4,
    "dropout": 0.1,
    "input_dim": 64,
    "output_dim": 3,
    "batch_first": True

}

learning_params = {
    # "lr": 8.906324028628413e-5,
    "lr": 1e-05,
    "epochs": 30,
    "weight_decay": 1e-4,
    "patience": 3,
    "lr_drop": 15,
}

dataloader_params = {
    
    "batch_size": 1,
    "one_hot": True,
    "observation_window": 30,
    "prediction_window": 10,
    "normalizer": '',  # ('standardization', 'min-max', 'power', ''),
    "base_path": '/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/EMS_Interventions/annotations/',
    "fold": 1,
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