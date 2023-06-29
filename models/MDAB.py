import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import Counter
import numpy as np
import pandas as pd
from ..utils.functions import ReverseLayerF


class FeatureExtractor(nn.Module):
    def __init__(self, first_conv_out=16):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=first_conv_out, kernel_size=6, stride=2
        )
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(first_conv_out)
        self.pool1 = nn.MaxPool1d(kernel_size=6, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=first_conv_out,
            out_channels=first_conv_out * 4,
            kernel_size=3,
            stride=2,
        )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(first_conv_out * 4)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv1d(
            in_channels=first_conv_out * 4,
            out_channels=first_conv_out * 8,
            kernel_size=2,
            stride=2,
        )
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(first_conv_out * 8)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.pool3(x)

        return x


class Class_classifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super(Class_classifier, self).__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.initialize_lin(self.fc1)
        self.initialize_lin(self.fc2)
        self.initialize_lin(self.fc3, bias=1 / num_classes)
        self.dropout = nn.Dropout(0.5)

    @staticmethod
    def initialize_lin(layer, bias=0):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DomainDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, num_domains):
        super(DomainDiscriminator, self).__init__()
        self.num_domains = num_domains
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_domains)
        self.initialize_lin(self.fc1)
        self.initialize_lin(self.fc2)
        self.initialize_lin(self.fc3, bias=1 / num_domains)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(1)

    @staticmethod
    def initialize_lin(layer, bias=0):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, bias)

    def forward(self, x, alpha=None):
        if alpha is not None:
            x = ReverseLayerF.apply(x, alpha)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class MDAB(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_dim,
        num_classes,
        num_domains,
        first_conv_out=16,
    ):
        super(MDAB, self).__init__()

        self.feature_extractor = FeatureExtractor(first_conv_out=first_conv_out)

        self.fc_input_size = self._get_fc_input_size(input_size)

        self.class_classifier = Class_classifier(
            self.fc_input_size, hidden_dim, num_classes
        )

        self.domain_discriminator = DomainDiscriminator(
            self.fc_input_size, hidden_dim, num_domains
        )

        # Define the loss functions
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        # Define the optimizer

    def _optimizer(self, lr=1e-3):
        optimizer = optim.SGD(
            [
                {"params": self.feature_extractor.parameters()},
                {"params": self.class_classifier.parameters()},
                {"params": self.domain_discriminator.parameters()},
            ],
            lr=lr,
        )
        return optimizer

    def _get_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size)
        x = self.feature_extractor(dummy_input)
        flattened_size = x.size(1) * x.size(2)
        return flattened_size

    def forward(self, x, alpha=None):
        x = torch.unsqueeze(x, 1)
        x = self.feature_extractor(x)
        features = x.view(x.size(0), -1)
        class_outputs = self.class_classifier(features)
        domain_outputs = self.domain_discriminator(features, alpha)
        return class_outputs, domain_outputs

    def _calculate_proportions(self, category_list):
        category_counts = Counter(category_list)
        total_count = len(category_list)
        proportions = [
            category_counts[category] / total_count for category in category_list
        ]
        proportions_dict = {
            category: category_counts[category] / total_count
            for category in set(category_list)
        }
        return proportions, proportions_dict

    def train_step(
        self,
        train_loader,
        optimizer = None, 
        epoch=None,
        num_epochs=None,
        domain_lambda=0.1,
        best_model=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loss = 0.0
        len_dataloader = len(train_loader)

        _, domains_proportions_dict = self._calculate_proportions(
            train_loader.dataset.domains.tolist()
        )

        for idx, (inputs, labels, domains, _) in enumerate(train_loader):
            p = float(idx + epoch * len_dataloader) / num_epochs / len_dataloader
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            domains_labels = domains.to(device)

            # Forward pass
            class_pred, domain_pred = self.forward(inputs, alpha)

            if torch.isnan(class_pred).any() and best_model is not None:
                print("\nclass_pred contains NA!")
                self.load_state_dict(best_model)
                break

            # Calculate the losses
            domains_labels = domains_labels.long()
            prediction_loss = self.class_criterion(class_pred, labels)
            domain_loss = self.domain_criterion(domain_pred, domains_labels)

            domain_probabilities_batch = F.softmax(domain_pred, dim=1)

            # Calculate the importance weights
            # domains_true_probabilities = torch.FloatTensor(self._calculate_proportions(domains_labels)).to(device)

            domains_true_probabilities = torch.FloatTensor(
                [domains_proportions_dict[i] for i in domains.tolist()]
            ).to(device)
            domain_probabilities_batch_1d = torch.FloatTensor(
                [
                    domain_probabilities_batch[idx, i]
                    for idx, i in enumerate(domains_labels)
                ]
            ).to(device)

            importance_weights = 1 + (
                domains_true_probabilities / (1 - domains_true_probabilities)
            ) * ((1 - domain_probabilities_batch_1d) / domain_probabilities_batch_1d)

            # Update the loss with importance weighting
            total_loss = (
                torch.mean(importance_weights * prediction_loss)
                + domain_lambda * domain_loss
            )

            # total_loss = prediction_loss + 0.1 * domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * inputs.size(0)

        return train_loss

    def evaluate(self, data_loader, cal_auc=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the model to evaluation mode
        self.to(device)
        self.eval()

        with torch.no_grad():
            prediction_losses = []
            predicted_labels_list = []
            prediction_score_list = []
            true_labels_list = []
            id_list = []

            for _, (inputs, labels, domains, ids) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                domains = domains.to(device)

                # Forward pass
                class_outputs, _ = self.forward(inputs)

                # Calculate the prediction loss for the batch
                prediction_loss = self.class_criterion(class_outputs, labels)
                prediction_losses.append(prediction_loss.item())

                # Store predicted and true labels for calculation of classification accuracy
                _, predicted = torch.max(class_outputs, dim=1)
                predicted_labels_list.extend(predicted.cpu().tolist())
                true_labels_list.extend(labels.cpu().numpy())

                class_probabilities = (
                    torch.softmax(class_outputs, dim=1)[:, 1].cpu().tolist()
                )
                prediction_score_list.extend(class_probabilities)
                id_list.extend(ids)

            classification_accuracy = accuracy_score(
                true_labels_list, predicted_labels_list
            )

            if cal_auc:
                auc_score = roc_auc_score(true_labels_list, prediction_score_list)
            else:
                auc_score = None

            average_prediction_loss = sum(prediction_losses) / len(prediction_losses)

        return {
            "results": (average_prediction_loss, classification_accuracy, auc_score),
            "predictions": prediction_score_list,
            "true_labels": true_labels_list,
            "ids": id_list,
        }
