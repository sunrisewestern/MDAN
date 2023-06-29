import torch
import torch.nn as nn
from .functions import ReverseLayerF

class DomainAdversarialNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, num_domains):
        super(DomainAdversarialNetwork, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=6, stride=2),

            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc_input_size = self._get_fc_input_size(input_size)
        
        self.class_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_domains),
            nn.Sigmoid()
        )
        
    def _get_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size)
        x = self.conv_layers(dummy_input)
        flattened_size = x.size(1) * x.size(2)
        return flattened_size
    
    def forward(self, x, alpha=None):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        features = x
        # features = self.additional_layers(x)

        if alpha is not None:
            reverse_features = ReverseLayerF.apply(features, alpha)
            domain_pred = self.domain_classifier(reverse_features)
        else:
            domain_pred = self.domain_classifier(features)

        class_pred = self.class_classifier(features)

        return class_pred, domain_pred
    

class MADA(nn.Module):
    '''
    Multi-Adversarial Domain Adaptation (https://arxiv.org/abs/1809.02176).
    For multi-label classification problem
    '''
    def __init__(self, input_size, hidden_dim, output_dim, num_domains):
        super(MADA, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=6, stride=2),

            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc_input_size = self._get_fc_input_size(input_size)
        
        self.class_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_domains),
            nn.Softmax()
        )
        
        
    def _get_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size)
        x = self.conv_layers(dummy_input)
        flattened_size = x.size(1) * x.size(2)
        return flattened_size
    
    def forward(self, x, alpha=None):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        class_logits = self.class_classifier(features)
        domain_logits = []
        
        if alpha is not None:
            reverse_features = ReverseLayerF.apply(features, alpha)
        else:
            reverse_features = features
        
        class_pred = nn.functional.softmax(class_logits, dim=1)
        for class_idx in range(self.num_classes):
            weighted_reverse_features = class_pred[:, class_idx].unsqueeze(1) * reverse_features
            domain_logits.append(
				self.domain_classifier(weighted_reverse_features).cuda()
			)

        return class_logits, domain_logits