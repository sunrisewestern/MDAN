import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        
    def evaluate(self, data_loader):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        test_predictions = []
        test_true_labels = []

        with torch.no_grad():
            for inputs, labels, _ in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                class_pred, _ = self.model(inputs)

                # Compute loss
                loss = nn.CrossEntropyLoss()(class_pred, labels)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(class_pred, 1)
                correct += (predicted == labels).sum().item()

                # Collect predictions and true labels for AUC calculation
                test_predictions.extend(torch.softmax(class_pred, dim=1)[:, 1].cpu().tolist())
                test_true_labels.extend(labels.cpu().tolist())
                
        # Calculate average test loss and accuracy
        test_loss /= len(data_loader.dataset)
        accuracy = correct / len(data_loader.dataset)

        # Calculate AUC
        try:
            test_auc = roc_auc_score(test_true_labels, test_predictions)
        except ValueError:
            test_auc = None

        return {
            'results' : (test_loss, accuracy, test_auc),
            'predictions' : test_predictions,
            'true_labels' :  test_true_labels
        }
        
def evaluate_model(model, data_loader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels, domains in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            class_pred, _ = model(inputs)

            # Compute loss
            loss = nn.CrossEntropyLoss()(class_pred, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(class_pred, 1)
            correct += (predicted == labels).sum().item()

            # Collect predictions and true labels for AUC calculation
            predictions.extend(torch.softmax(class_pred, dim=1)[:, 1].cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    # Calculate average test loss and accuracy
    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    # Calculate AUC
    auc = roc_auc_score(true_labels, predictions)

    return {
            'results' : (test_loss, accuracy, auc),
            'predictions' : predictions,
            'true_labels' :  true_labels
        }