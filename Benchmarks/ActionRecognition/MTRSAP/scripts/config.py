import torch


RECORD_RESULTS = True

tcn_model_params = {
    "encoder_params": { #some of these gets updated during runtime based on the feature dimension of the given data
        "in_channels": 14,
        "kernel_size": 31,
        "out_channels": 64,
    },
    "decoder_params": {
        "in_channels": 64,
        "kernel_size": 31,
        "out_channels": 64
    }
}


transformer_params = {
    "d_model": 64,
    "nhead": 2,
    "num_layers": 2,
    "hidden_dim": 128,
    "layer_dim": 4,
    "dropout": 0.1,
    "input_dim": 64,
    "output_dim": 7,
    "batch_first": True

}

learning_params = {
    # "lr": 8.906324028628413e-5,
    "lr": 1e-05,
    "epochs": 20,
    "weight_decay": 1e-4,
    "patience": 3
}

dataloader_params = {
    
    "batch_size": 10,
    "one_hot": True,
    "observation_window": 30,
    "prediction_window": 10,
    "user_left_out": 7,
    "cast": True,
    "include_image_features": False,
    "normalizer": '',  # ('standardization', 'min-max', 'power', '')
    "step": 1,  # 1 - 30 Hz
    "context": 9  # 0-nocontext, 1-contextonly, 2-context+kin, 3-imageonly, 4-image+kin, 5-image+kin+context, 6-colin_features, 7- colin+context, 8-colin+kin, 9-colin+kin+context, 10-segonly, 11-kin+seg, 12-kin+seg+context, 13-kin+seg+context+colins, 14-seg+colins
    # hamid -  do not need (1,3,5,7)
    
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