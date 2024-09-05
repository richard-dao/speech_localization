import torch
import torch.nn as nn
import snntorch as snn
from .Neurons.TernaryLeaky import TernaryLeaky
from config import config

class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()

        self.dropout_rates = config["model_config"]["dropout_rates"]
        self.betas = config["model_config"]["betas"]
        self.thresholds = config["model_config"]["thresholds"]
        self.num_features, self.feature_height, self.feature_width = self.determine_feature_dimensions()
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(self.num_features, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout_rates[0]),
            TernaryLeaky(beta=self.betas[0], threshold_positive=1.0, threshold_negative=-1.0, learn_beta=True, learn_threshold=True, init_hidden=True),
            

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout_rates[1]),
            TernaryLeaky(beta=self.betas[1], threshold_positive=1.0, threshold_negative=-1.0, learn_beta=True, learn_threshold=True, init_hidden=True),
            

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout_rates[2]),
            TernaryLeaky(beta=self.betas[2], threshold_positive=1.0, threshold_negative=-1.0, learn_beta=True, learn_threshold=True, init_hidden=True),
            

            nn.Flatten()
        )

        self._init_feature_size()

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size//32),
            nn.BatchNorm1d(self.feature_size//32),
            nn.Dropout(self.dropout_rates[3]),
            TernaryLeaky(beta=self.betas[3], threshold_positive=1.0, threshold_negative=-1.0, learn_beta=True, learn_threshold=True, init_hidden=True),
            
            nn.Linear(self.feature_size//32, self.feature_size//64),
            nn.BatchNorm1d(self.feature_size//64),
            nn.Dropout(self.dropout_rates[4]),
            TernaryLeaky(beta=self.betas[4], threshold_positive=1.0, threshold_negative=-1.0, learn_beta=True, learn_threshold=True, init_hidden=True),
            
            nn.Linear(self.feature_size//64, self.feature_size//128),
            nn.BatchNorm1d(self.feature_size//128),
            nn.Dropout(self.dropout_rates[5]),
            TernaryLeaky(beta=self.betas[5], threshold_positive=1.0, threshold_negative=-1.0, learn_beta=True, learn_threshold=True, init_hidden=True),
            
            nn.Linear(self.feature_size//128, self.feature_size//256),
            nn.BatchNorm1d(self.feature_size//256),
            nn.Dropout(self.dropout_rates[6]),
            TernaryLeaky(beta=self.betas[6], threshold_positive=1.0, threshold_negative=-1.0, learn_beta=True, learn_threshold=True, init_hidden=True),
            
            nn.Linear(self.feature_size//256, self.feature_size//512),
            nn.BatchNorm1d(self.feature_size//512),
            #nn.Dropout(self.dropout_rates[7]),
            TernaryLeaky(beta=self.betas[7], threshold_positive=1.0, threshold_negative=-1.0, learn_beta=True, learn_threshold=True, init_hidden=True),
            nn.Linear(self.feature_size//512, 2)  # Output layer for azimuth and elevation
        )

        self.initialize_weights()

    def _init_feature_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_features, self.feature_height, self.feature_width)
            dummy_features = self.features(dummy_input)
            self.feature_size = dummy_features.shape[-1]

    def determine_feature_dimensions(self):
        feature_dims = {
            "log_mel_spectrogram": (4, 126, 128),
            "gcc_phat": (6, 512, 126),
            "hilbert_transform": (4, 512, 124),
            "active_reactive_intensities": (6, 257, 126),
            "gcc_hilbert": (10, 512, 124)
        }
        selected_feature = config["model_config"]["features"]
        return feature_dims.get(selected_feature, (1, 64, 64))

    def forward(self, x):
        if x.dim() == 5:  # If there is a time step dimension
            outputs = []
            for t in range(x.size(1)):  # Iterate over time steps
                step_input = x[:, t, :, :, :]
                features = self.features(step_input)
                output = self.classifier(features)
                outputs.append(output)
            outputs = torch.stack(outputs, dim=1)
            return torch.mean(outputs, dim=1)
        else:  # If there is no time step dimension
            features = self.features(x)
            return self.classifier(features)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

