import torch
from data import resnet_features,kinematic_feature_names,colin_features, segmentation_features, kinematic_feature_names_jigsaws, kinematic_feature_names_jigsaws_patient_position, class_names, all_class_names, state_variables


RECORD_RESULTS = True

tcn_model_params = {
    "class_num": 7,
    "decoder_params": {
        "input_size": 256,
        "kernel_size": 61,
        "layer_sizes": [
            96,
            64,
            # 64
        ],
        "layer_type": "TempConv",
        "norm_type": "Channel",
        "transposed_conv": True
    },
    "encoder_params": {
        "input_size": 25,
        "kernel_size": 61,
        "layer_sizes": [
            64,
            256,
            # 128
        ],
        "layer_type": "TempConv",
        "norm_type": "Channel"
    },
    "fc_size": 32,
    "mid_lstm_params": {
        "hidden_size": 64,
        "input_size": 256,
        "layer_num": 1
    }
}


transformer_params = {
    "d_model": 64,
    "nhead": 1,
    "num_layers": 1,
    "hidden_dim": 128,
    "layer_dim": 4,
    "encoder_params": { #some of these gets updated during runtime based on the feature dimension of the given data
        "in_channels": 14,
        "kernel_size": 31,
        "out_channels": 64,
    },
    "decoder_params": {
        "in_channels": 64,
        "kernel_size": 31,
        "out_channels": 64
    },
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
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
        
        # Model parameters
        self.tcn_model_params = tcn_model_params
        self.transformer_params = transformer_params

        # Learning parameters
        self.learning_params = learning_params

        # DataLoader parameters
        self.dataloader_params = dataloader_params