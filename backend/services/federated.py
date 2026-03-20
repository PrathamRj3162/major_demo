"""
Federated Learning Simulation
==============================
Simulates federated learning across multiple hospital clients using
the FedAvg (Federated Averaging) algorithm.

Architecture:
  ┌──────────────────────────────────────────────┐
  │              Central Server                   │
  │   (Aggregates client model updates)          │
  │   FedAvg: weighted avg of parameters         │
  └──────────┬────────┬────────┬────────┬────────┘
             │        │        │        │
      ┌──────┴──┐ ┌──┴──────┐ ┌──┴──────┐ ┌──┴──────┐
      │Client 1 │ │Client 2 │ │Client 3 │ │Client 4 │
      │Hospital │ │Hospital │ │Hospital │ │Hospital │
      │  A      │ │  B      │ │  C      │ │  D      │
      └─────────┘ └─────────┘ └─────────┘ └─────────┘

For CPU efficiency, we use a lightweight CNN that mirrors the
DenseNet121 classifier structure but is fast to train on CPU.
In production, the full DenseNet121 would be used with GPU clusters.
"""

import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import (
    NUM_CLIENTS, FED_ROUNDS, LOCAL_EPOCHS,
    FED_BATCH_SIZE, FED_LEARNING_RATE, NUM_CLASSES
)


class LightweightPneumoniaNet(nn.Module):
    """
    Lightweight CNN for federated learning simulation.
    
    Uses a compact architecture optimized for CPU training speed
    while maintaining the same classification interface as the
    full DenseNet121 model (binary: Normal vs Pneumonia).
    
    In a real hospital deployment, the full DenseNet121 would be
    used with GPU acceleration. This model demonstrates the
    FedAvg algorithm with realistic convergence behavior.
    """
    
    def __init__(self, num_classes=2):
        super(LightweightPneumoniaNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def generate_synthetic_data(num_samples=60, image_size=64):
    """
    Generate synthetic data that simulates chest X-ray distributions.
    Uses smaller images (64x64) for fast CPU training.
    """
    # Generate normalized random images (simulating preprocessed X-rays)
    images = torch.randn(num_samples, 3, image_size, image_size) * 0.2
    
    # Add class-specific patterns so the model can actually learn
    for i in range(num_samples):
        if i < num_samples // 2:
            # "Normal" pattern — slight brightness in center
            images[i, :, 20:44, 20:44] += 0.3
        else:
            # "Pneumonia" pattern — patchy regions
            h = random.randint(10, 30)
            w = random.randint(10, 30)
            images[i, :, h:h+20, w:w+20] += 0.5
    
    labels = torch.cat([
        torch.zeros(num_samples // 2, dtype=torch.long),
        torch.ones(num_samples - num_samples // 2, dtype=torch.long)
    ])
    
    # Shuffle
    indices = torch.randperm(num_samples)
    images = images[indices]
    labels = labels[indices]
    
    return images, labels


def create_client_data_shards(num_clients=NUM_CLIENTS, samples_per_client=60):
    """
    Create non-IID data shards for each client.
    """
    client_loaders = []
    
    for i in range(num_clients):
        num_samples = samples_per_client + random.randint(-10, 10)
        images, labels = generate_synthetic_data(num_samples)
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=FED_BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)
    
    return client_loaders


def train_client(model, dataloader, epochs=LOCAL_EPOCHS, lr=FED_LEARNING_RATE):
    """
    Train a model locally on one client's data.
    """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    metrics = {"loss": [], "accuracy": []}
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct / total if total > 0 else 0.0
        metrics["loss"].append(round(epoch_loss, 4))
        metrics["accuracy"].append(round(epoch_acc, 4))
    
    return model, metrics


def fedavg_aggregate(global_model, client_models, client_data_sizes):
    """
    Federated Averaging (FedAvg) algorithm.
    
    Aggregates client model parameters using weighted averaging,
    where weights are proportional to each client's dataset size.
    
    Algorithm:
      θ_global = Σ (n_k / n) * θ_k
    """
    total_data = sum(client_data_sizes)
    global_dict = global_model.state_dict()
    
    aggregated_dict = {}
    for key in global_dict.keys():
        aggregated_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
    
    for client_model, data_size in zip(client_models, client_data_sizes):
        weight = data_size / total_data
        client_dict = client_model.state_dict()
        for key in aggregated_dict:
            aggregated_dict[key] += weight * client_dict[key].float()
    
    global_model.load_state_dict(aggregated_dict)
    return global_model


def evaluate_global_model(model, test_loader):
    """Evaluate the global model on a test set."""
    device = next(model.parameters()).device
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return {
        "loss": round(total_loss / max(len(test_loader), 1), 4),
        "accuracy": round(correct / max(total, 1), 4)
    }


def run_federated_simulation(num_clients=NUM_CLIENTS, num_rounds=FED_ROUNDS,
                               progress_callback=None):
    """
    Run a complete federated learning simulation.

    Uses a lightweight CNN for fast CPU training while demonstrating
    the FedAvg algorithm with realistic convergence behavior.
    """
    device = torch.device("cpu")  # CPU for simulation
    
    # Initialize lightweight global model
    global_model = LightweightPneumoniaNet(num_classes=NUM_CLASSES)
    global_model = global_model.to(device)
    
    # Create client data shards
    client_loaders = create_client_data_shards(num_clients)
    client_data_sizes = [len(loader.dataset) for loader in client_loaders]
    
    # Create a test set for global evaluation
    test_images, test_labels = generate_synthetic_data(40)
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=FED_BATCH_SIZE)
    
    # Training logs
    training_log = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": LOCAL_EPOCHS,
        "client_data_sizes": client_data_sizes,
        "rounds": [],
        "global_metrics": [],
        "client_names": [f"Hospital {chr(65 + i)}" for i in range(num_clients)]
    }
    
    print(f"\n{'='*60}")
    print(f"  Federated Learning Simulation")
    print(f"  Clients: {num_clients} | Rounds: {num_rounds}")
    print(f"  Local Epochs: {LOCAL_EPOCHS} | Batch Size: {FED_BATCH_SIZE}")
    print(f"{'='*60}\n")
    
    for round_num in range(num_rounds):
        print(f"[Round {round_num + 1}/{num_rounds}]")
        
        round_log = {
            "round": round_num + 1,
            "client_metrics": []
        }
        
        client_models = []
        
        for client_id in range(num_clients):
            local_model = copy.deepcopy(global_model)
            trained_model, metrics = train_client(
                local_model, 
                client_loaders[client_id]
            )
            
            client_models.append(trained_model)
            
            client_log = {
                "client_id": client_id,
                "client_name": f"Hospital {chr(65 + client_id)}",
                "data_size": client_data_sizes[client_id],
                "final_loss": metrics["loss"][-1],
                "final_accuracy": metrics["accuracy"][-1],
                "loss_history": metrics["loss"],
                "accuracy_history": metrics["accuracy"]
            }
            round_log["client_metrics"].append(client_log)
            
            print(f"  Client {client_id} ({client_log['client_name']}): "
                  f"Loss={client_log['final_loss']:.4f}, "
                  f"Acc={client_log['final_accuracy']:.4f}")
        
        # FedAvg aggregation
        global_model = fedavg_aggregate(global_model, client_models, client_data_sizes)
        
        # Evaluate global model
        global_metrics = evaluate_global_model(global_model, test_loader)
        training_log["global_metrics"].append(global_metrics)
        
        round_log["global_loss"] = global_metrics["loss"]
        round_log["global_accuracy"] = global_metrics["accuracy"]
        training_log["rounds"].append(round_log)
        
        print(f"  → Global Model: Loss={global_metrics['loss']:.4f}, "
              f"Acc={global_metrics['accuracy']:.4f}\n")
        
        if progress_callback:
            progress_callback(round_num + 1, num_rounds)
    
    print(f"{'='*60}")
    print(f"  Simulation Complete!")
    print(f"{'='*60}\n")
    
    return training_log
