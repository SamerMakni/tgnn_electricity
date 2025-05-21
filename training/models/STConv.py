import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv
import parameters

from progress_table import ProgressTable
import time 
import os 
import pickle as pkl
class AttentionGCN(nn.Module):
    def __init__(self, hidden_size, num_nodes, num_features):
        super(AttentionGCN, self).__init__()
                      
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.kernel_size = hidden_size
        self.pad_size = self.kernel_size - 1

        self.gnn = STConv(
            num_nodes=self.num_nodes, 
            in_channels=1,
            hidden_channels=32, 
            out_channels=32, 
            kernel_size=self.kernel_size, 
            K=1)

        self.conv = nn.Sequential( 
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(1, num_features),
                bias=True,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=1,
                kernel_size=(1, 1),
                bias=True,
            )
        )

    def forward(self, window, edge_index, edge_weight):
        # Prepare the input data
        x = F.pad(window.x, (self.pad_size, self.pad_size))
        x = x.view(-1, self.num_nodes, self.num_features + 2 * self.pad_size, 1).permute(0, 2, 1, 3).to(parameters.device)
   
        # Apply GNN and convolution
        H = F.relu(self.gnn(x, edge_index, edge_weight)) # -> (1, lags, num_nodes, out_channels)
        H = H.permute(0, 3, 2, 1)  # -> (1, out_channels, num_nodes, lags)
        x = self.conv(H).squeeze(3) # -> (1, num_nodes)
        
        # Flatten the output
        x = x.flatten()
        return x

def train(model, train_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)
    model.train()
    loss_history = []
    
    table = ProgressTable(
        pbar_embedded=False,
        pbar_style='angled alt red blue',
    )
    table.add_columns('Epoch')
    table.add_columns('Train MSE', color='blue')
    table.add_columns('Time', color='green')
    sample = next(iter(train_dataset))
    edge_index = sample.edge_index.to(parameters.device)
    edge_weight = sample.edge_attr.to(parameters.device)
    
    start_time = time.time()
    for epoch in table(parameters.epochs, show_throughput=False, show_eta=True, description="Training"):
        table['Epoch'] = f'{epoch}/{parameters.epochs}'
        total_loss = 0
        for snap in table(enumerate(train_dataset), total=len(train_dataset.features), description="Epoch"):
            i, window = snap
            optimizer.zero_grad()
            window = window.to(parameters.device)
            
            y_pred = model(window, edge_index, edge_weight)
            assert y_pred.shape == window.y.shape
            
            loss = torch.mean((y_pred - window.y) ** 2)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        elapsed_time = (time.time() - start_time)
        total_loss /= i + 1
        loss_history.append(total_loss)
        
        if epoch % 10 == 0:
            table.update("Train MSE", total_loss, color="blue")
            table.update("Time", f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}", color="green")
            table.next_row()
    
    table.close()
    return model, loss_history


def eval(model, test_dataset):
    model.eval()
    loss = 0
    sample = next(iter(test_dataset))
    edge_index = sample.edge_index.to(parameters.device)
    edge_weight = sample.edge_attr.to(parameters.device)
    with torch.no_grad():
        for i, window in enumerate(test_dataset):
            window.to(parameters.device)
            y_pred = model(window, edge_index, edge_weight)
            assert y_pred.shape == window.y.shape
            loss += torch.mean((y_pred - window.y) ** 2)
        loss /= i + 1
    
    mse = loss.item()
    absolute_mean_error = torch.mean(torch.abs(y_pred - window.y)).item()
    r2 = 1 - (torch.sum((y_pred - window.y) ** 2) / torch.sum((window.y - torch.mean(window.y)) ** 2)).item()
    return mse, absolute_mean_error, r2


def report(model, test_dataset, loss_history, model_name):
    fig, ax = plt.subplots()

    x_ticks = np.arange(1, parameters.epochs + 1)
    ax.plot(x_ticks, loss_history)

    ax.set_title('Loss over time', fontweight='bold')
    ax.set_xlabel('Epochs', fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontweight='bold')
    
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(f'{parameters.results_location}/{model_name}_{current_time}', exist_ok=True)
    plt.savefig(f'{parameters.results_location}/{model_name}_{current_time}/loss.png')
    sample = next(iter(test_dataset))
    edge_index = sample.edge_index.to(parameters.device)
    edge_weight = sample.edge_attr.to(parameters.device)
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        for i, window in enumerate(test_dataset):
            window.to(parameters.device)
            y_pred = model(window, edge_index, edge_weight)
            actuals.append(window.y.cpu().numpy())
            predictions.append(y_pred.cpu().numpy())

    actuals = np.concatenate(actuals, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    os.makedirs(f'{parameters.results_location}/{model_name}_{current_time}', exist_ok=True)
    plt.savefig(f'{parameters.results_location}/{model_name}_{current_time}/scatter.png')
    plt.show()

    # Plot per node
    num_nodes = 37
    time_steps = actuals.shape[0] // num_nodes

    actuals = actuals.reshape(time_steps, num_nodes)
    predictions = predictions.reshape(time_steps, num_nodes)

    rows = int(np.ceil(np.sqrt(num_nodes)))
    cols = rows if rows * (rows - 1) < num_nodes else rows - 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()

    for node in range(num_nodes):
        ax = axes[node]
        ax.plot(range(time_steps), actuals[:, node], color='blue')
        ax.plot(range(time_steps), predictions[:, node], color='orange')
        country_name = parameters.country_borders[node][1]
        ax.set_title(f'{country_name}')

    for i in range(num_nodes, len(axes)):
        fig.delaxes(axes[i])

    axes[0].plot([], [], color='blue', label='Actual')
    axes[0].plot([], [], color='orange', label='Predicted')

    fig.legend(loc='lower right', fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    os.makedirs(f'{parameters.results_location}/{model_name}_{current_time}', exist_ok=True)
    plt.savefig(f'{parameters.results_location}/{model_name}_{current_time}/predictions.png')
    plt.show()
    pickle.dump(model, open(f'{parameters.results_location}/{model_name}_{current_time}/model.pkl', 'wb'))