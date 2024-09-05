import torch


config = {
    "env":{
        "ip": "", # enter internal IP
        "port": "5000"}, # enter port
    
    "paths": {
        "metadata_path": "/mnt/persistent/metadata.parquet",
        "speech_path": "/mnt/persistent/ambisonics",
        "noise_path": "/mnt/persistent/noise",
        "save_dir": "../data/preprocessed",
        "model_path": "/home/ubuntu/Desktop/cnn_gcc_hilbert_20_46.pth", # cnn_gcc_hilbert_20_46.pth | crnn_gcc_hilbert_21_37.pth | snn_gcc_hilbert_32_66 | srnn_gcc_hilbert_32_58.pth
    },
    
    
    "audio_processing": {
        "sr": 16000,
        "start": 0,
        "duration": 2,
        "noise_ratio": 1,
        "n_fft": 512,
        "stht_frames": 512,
        "hop_length": 256,
        "mel_bins": 128,
        "f_min": 50,
        "window": "hann",
    },
    
    
    "model_config": {
        "batch_size": 16,
        "num_workers": 16,
        "use_snn": True,
        "model_type": "SNN", # "CNN", "CRNN", "SNN", "SRNN"
        "num_steps": 4,
        "dropout_rates": [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.3, 0.2], # [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.3, 0.2]
        "thresholds": [1.0] * 8,
        "betas": [0.9] * 8,
        "features": "gcc_hilbert", # "hilbert_transform" | "log_mel_spectrogram" | "gcc_phat" | "active_reactive_intensities" | "gcc_hilbert"
        "encoding_type": "rate", # "rate" | None   
    },
    
    
    "training": {
        "is_lite": True,
        "num_epochs": 30,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "val_size": 0.2,
        "learning_rate": 0.003,
        "l2": 1e-4,
        
        "lr_scheduler": {
            "t_max": 20,
            "eta_min": 1e-7
        },
        
        "early_stopping": {
            "patience": 7,
            "delta": 0.05
        },
        
        "random_state": 11,
    },
}
