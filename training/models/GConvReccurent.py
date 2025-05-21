import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU
import parameters

from progress_table import ProgressTable
import time 
import pickle as pkl
import os

# class RecurrentGCN(nn.Module):
#     def __init__(self, cell, hidden_size ,node_features, window_length):
#         super(RecurrentGCN, self).__init__()
#         self.node_features = node_features
#         self.n = window_length
#         self.cell = cell
#         self.hidden_state_size = hidden_size
#         assert self.hidden_state_size % 2 == 0
        
        
#         self.recurrent_1 = cell(node_features, self.hidden_state_size, 2) # 37 -> 512    
#         self.recurrent_2 = cell(self.hidden_state_size//2, self.hidden_state_size, 2) # 512 -> 512
#         self.linear1 = nn.Sequential(
#             nn.Linear(self.hidden_state_size, self.hidden_state_size//2), # 512 -> 256
#             nn.Dropout(p=0.2),
#             nn.ReLU()
#         )
#         self.out = nn.Sequential(
#             nn.Linear(self.hidden_state_size//2, 1), # 256 -> 1
#             nn.Flatten(start_dim=0, end_dim=-1)
#         )
# def forward(self, window, h=None, c=None):        
    #     edge_index = window.edge_index
    #     edge_weight = None
    
    #     H, C = [], []
        
    #     x = window.x 
    #     if issubclass(self.cell, GConvLSTM):
    #         h, c = self.recurrent_1(x, edge_index, edge_weight, h, c)
    #         H.append(h.detach())
    #         C.append(c.detach())
    #         x = self.linear1(h)
    #         h, c = self.recurrent_2(x, edge_index, edge_weight, h, c)
    #         H.append(h.detach())
    #         C.append(c.detach())
    #         x = self.linear1(h)
    #     elif issubclass(self.cell, GConvGRU):
    #         h, c = self.recurrent_1(x, edge_index, edge_weight, h, c)
    #         H.append(h.detach())
    #         C.append(c.detach())
    #         x = self.linear1(h)
    #         h, c = self.recurrent_2(x, edge_index, edge_weight, c)
    #         H.append(h.detach())
    #         C.append(c.detach())
    #         x = self.linear1(h)
    #     pred = self.out(x)

    #     return pred, H, C
    
class RecurrentGCN(nn.Module):
    def __init__(self, cell, hidden_size, node_features, window_length):
        super(RecurrentGCN, self).__init__()
        self.node_features = node_features
        self.n = window_length
        self.cell = cell
        self.hidden_state_size = hidden_size
        assert self.hidden_state_size % 2 == 0
        
        self.recurrent = cell(node_features, self.hidden_state_size, 2)
        
        # Making the model deeper by adding additional layers
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size),  # first deeper layer
            nn.ReLU(),
            nn.Linear(self.hidden_state_size, self.hidden_state_size//2),  # second deeper layer
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        
        self.linear2 = nn.Sequential(
            nn.Linear(self.hidden_state_size//2, self.hidden_state_size//4),  # third deeper layer
            nn.ReLU(),
            nn.Linear(self.hidden_state_size//4, self.hidden_state_size//2),  # fourth deeper layer
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        
        # Final output projection layer to ensure the output dimension is 1
        self.out = nn.Sequential(
            nn.Linear(self.hidden_state_size//2, 1),
            nn.Flatten(start_dim=0, end_dim=-1)
        )

    def forward(self, window, h=None, c=None):        
        edge_index = window.edge_index
        edge_weight = None
    
        H, C = [], []
        
        x = window.x 
        if issubclass(self.cell, GConvLSTM):
            h, c = self.recurrent(x, edge_index, edge_weight, h, c)
            H.append(h.detach())
            C.append(c.detach())
            x = self.linear1(h)  
            x = self.linear2(x)  
        elif issubclass(self.cell, GConvGRU):
            c = self.recurrent(x, edge_index, edge_weight, c)
            C.append(c.detach())
            x = self.linear1(c)
            x = self.linear2(x)
        
        pred = self.out(x)

        return pred, H, C
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def train(model, train_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)
    model.train()
    loss_history = []
    
    table = ProgressTable(
        pbar_embedded=False,
        pbar_style='angled alt red blue',
    )
    table.add_columns('Epoch')
    table.add_columns('Train MSE', color = 'blue')
    table.add_columns('Time', color = 'green')
    start_time = time.time()
    for epoch in table(parameters.epochs, show_throughput=False, show_eta=True, description="Training"):
        table['Epoch'] = f'{epoch}/{parameters.epochs}'
        h, c = None, None
        total_loss = 0
        for snap in table(enumerate(train_dataset), total=len(train_dataset.features) ,description="Epoch"):
            i, window = snap
            optimizer.zero_grad()
            window = window.to(parameters.device)
            if issubclass(model.cell, GConvLSTM):
                y_pred, H, C = model(window, c, h)
                h = H[parameters.stride-1].to(parameters.device)
                c = C[parameters.stride-1].to(parameters.device)
            elif issubclass(model.cell, GConvGRU):
                y_pred, _ ,C = model(window, c)
                c = C[parameters.stride-1].to(parameters.device)

            assert y_pred.shape == window.y.shape
            loss = torch.mean((y_pred - window.y)**2)
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
    return model, loss_history, elapsed_time
    
    
def eval(model, test_dataset):
    model.eval()
    loss = 0
    with torch.no_grad():
        h, c = None, None
        for i, window in enumerate(test_dataset):
            window.to(parameters.device)
            if issubclass(model.cell, GConvLSTM):
                y_pred, H, C = model(window, c, h)
                h = H[parameters.stride-1].to(parameters.device)
                c = C[parameters.stride-1].to(parameters.device)
            elif issubclass(model.cell, GConvGRU):
                y_pred, _ ,C = model(window, c)
                c = C[parameters.stride-1].to(parameters.device)

            assert y_pred.shape == window.y.shap
            if parameters.normalizer == 'population':
                populations = torch.tensor([parameters.country_borders[i][2] for i in range(parameters.num_nodes)]).to(parameters.device)
                y_pred = y_pred * populations
                window.y = window.y * populations
            loss += torch.mean((y_pred - window.y)**2)
        loss /= i+1
    mse = loss.item()
    absolute_mean_error = torch.mean(torch.abs(y_pred - window.y)).item()
    r2 = 1 - (torch.sum((y_pred - window.y)**2) / torch.sum((window.y - torch.mean(window.y))**2)).item()
    return mse, absolute_mean_error, r2

def report(model, test_dataset, loss_history, model_name):
    fig, ax = plt.subplots()

    x_ticks = np.arange(1, parameters.epochs+1)
    ax.plot(x_ticks, loss_history)

    ax.set_title('Loss over time', fontweight='bold')
    ax.set_xlabel('Epochs', fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontweight='bold')
    
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(f'{parameters.results_location}/{model_name}_{current_time}', exist_ok=True)
    plt.savefig(f'{parameters.results_location}/{model_name}_{current_time}/loss.png')
    
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        h, c = None, None
        for i, window in enumerate(test_dataset):
            window.to(parameters.device)
            if issubclass(model.cell, GConvLSTM):
                y_pred, H, C = model(window, c, h)
                h = H[parameters.stride-1].to(parameters.device)
                c = C[parameters.stride-1].to(parameters.device)
            elif issubclass(model.cell, GConvGRU):
                y_pred, _ ,C = model(window, c)
                c = C[parameters.stride-1].to(parameters.device)
            if parameters.normalizer == 'population':
                populations = torch.tensor([parameters.country_borders[i][2] for i in range(parameters.num_nodes)]).to(parameters.device)
                y_pred = y_pred * populations
                window.y = window.y * populations
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


    num_nodes = parameters.num_nodes
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
    pkl.dump(model, open(f'{parameters.results_location}/{model_name}_{current_time}/model.pkl', 'wb'))
    
        
        