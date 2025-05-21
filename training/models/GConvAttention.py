import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import parameters
from torch_geometric_temporal.nn.attention import ASTGCN
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score


from progress_table import ProgressTable
import time 
import pickle as pkl
import os


class AttentionGCN(nn.Module):
    def __init__(self, cell, hidden_size, node_features, num_nodes):
        super(AttentionGCN, self).__init__()
        self.node_features = node_features
        self.hidden_size = hidden_size
        self.cell = cell
        
                # Define dropout layer
        self.dropout = nn.Dropout(0.3)
        
        self.attention = ASTGCN(
            nb_block=3,
            in_channels=1,
            K=5,
            nb_chev_filter=16,
            nb_time_filter=16,
            time_strides=1,
            num_for_predict=1,
            len_input=self.node_features,
            num_of_vertices=parameters.num_nodes,
            normalization='sym',
            bias=True,
        )

    def forward(self, window, edge_index):     
        x = window.x.view(-1, parameters.num_nodes, 1, self.node_features)    
        x = self.attention(x, edge_index)        
        return x.squeeze(1).permute(0,2,1).flatten()
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def train(model, train_loader, validation_loader):   
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr, weight_decay=1e-4)  # L2 regularization
    model.train()
    loss_history = []
    edge_index = train_loader.dataset[0].edge_index.to(parameters.device)
    table = ProgressTable(
        pbar_embedded=False,
        pbar_style='angled alt red blue',
    )
    table.add_columns('Epoch')
    table.add_columns('Train MSE', color='blue')
    table.add_columns('Time', color='green')

    best_mse = float('inf')
    best_model_state = None

    start_time = time.time()

    for epoch in table(parameters.epochs, show_throughput=False, show_eta=True, description="Training"):
        table['Epoch'] = f'{epoch}/{parameters.epochs}'
        total_loss = 0
        
        for i, window in table(enumerate(train_loader), total=len(train_loader), description="Epoch"):
            optimizer.zero_grad()
            window = window.to(parameters.device)
            y_pred = model(window, edge_index)
            
            assert y_pred.shape == window.y.shape
            
            loss = torch.mean((y_pred - window.y) ** 2)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        elapsed_time = time.time() - start_time
        average_loss = total_loss / (i + 1)
        loss_history.append(average_loss)


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_window in validation_loader:
                val_window = val_window.to(parameters.device)
                val_pred = model(val_window, edge_index)
                val_loss += torch.mean((val_pred - val_window.y) ** 2).item()

        val_loss /= len(validation_loader)
        
        # Early stopping logic
        if val_loss < best_mse:
            best_mse = val_loss
            best_model_state = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= 20:
            print(f"Early stopping at epoch {epoch}.")
            parameters.epochs = epoch + 1
            break

        if epoch % 10 == 0:
            table.update("Train MSE", average_loss, color="blue")
            table.update("Time", f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}", color="green")
            table.next_row()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    table.close()
    return model, loss_history, elapsed_time

def eval(model, test_loader):
    model.eval()
    test_mse = 0
    test_mae = 0
    test_r2 = 0
    total_samples = 0

    edge_index = test_loader.dataset[0].edge_index.to(parameters.device)
    for i, window in enumerate(test_loader):
        window = window.to(parameters.device)
        y_pred = model(window, edge_index)
        
        total_samples += window.y.numel()
        if parameters.normalize:
            if parameters.normalizer == 'population':
                populations = torch.tensor([parameters.country_borders[i][2] for i in range(parameters.num_nodes)]).to(parameters.device)
                populations = populations.repeat(window.x.shape[0]//parameters.num_nodes)
                window.y = window.y * populations
                y_pred = y_pred * populations
                test_mae += torch.sum(torch.abs(y_pred - window.y)).item()
                test_mse += torch.sum((y_pred - window.y) ** 2).item()
                test_r2 += 1 - (torch.sum((y_pred - window.y) ** 2) / torch.sum((window.y - torch.mean(window.y)) ** 2)).item()
            if parameters.normalizer == 'std':
                std = torch.tensor([parameters.country_borders[i][3] for i in range(parameters.num_nodes)]).to(parameters.device)
                std = std.repeat(window.x.shape[0]//parameters.num_nodes)
                window.y = window.y * std
                y_pred = y_pred * std
                test_mae += torch.sum(torch.abs(y_pred - window.y)).item()
                test_mse += torch.sum((y_pred - window.y) ** 2).item()
                test_r2 += 1 - (torch.sum((y_pred - window.y) ** 2) / torch.sum((window.y - torch.mean(window.y)) ** 2)).item()
        else:            
            test_mse += torch.sum((y_pred - window.y) ** 2).item()
            test_mae += torch.sum(torch.abs(y_pred - window.y)).item()
            test_r2 += 1 - (torch.sum((y_pred - window.y) ** 2) / torch.sum((window.y - torch.mean(window.y)) ** 2)).item()
        
    test_mse /= total_samples
    test_mae /= total_samples
    test_r2 /= i + 1

    return test_mse, test_mae, test_r2


def report(model, test_dataset, loss_history, model_name):
    fig, ax = plt.subplots()
    edge_index = test_dataset.dataset[0].edge_index.to(parameters.device)
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
        for _, window in enumerate(test_dataset):
            window.to(parameters.device)
            y_pred = model(window, edge_index)
            if parameters.normalize:
                if parameters.normalizer == 'population':
                    populations = torch.tensor([parameters.country_borders[i][2] for i in range(parameters.num_nodes)]).to(parameters.device)
                    populations = populations.repeat(window.x.shape[0]//parameters.num_nodes)
                    window.y = window.y * populations
                    y_pred = y_pred * populations
                if parameters.normalizer == 'std':
                    std = torch.tensor([parameters.country_borders[i][3] for i in range(parameters.num_nodes)]).to(parameters.device)
                    std = std.repeat(window.x.shape[0]//parameters.num_nodes)
                    window.y = window.y * std
                    y_pred = y_pred * std
            
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

    fig, axes = plt.subplots(rows, cols, figsize=(32, 32))
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
    plt.savefig(f'{parameters.results_location}/{model_name}_{current_time}/predictions.png')    # ... existing visualization code ...

    # ================= Attention Matrix Plot =================
    try:
        model.eval()
        sample_window = test_dataset.dataset[0].to(parameters.device)
        edge_index = sample_window.edge_index.to(parameters.device)
        
        attention_maps = []

        def hook_function(module, input, output):
            attention_maps.append(output.detach().cpu().numpy())

        hooks = []
        for block in model.attention._blocklist:
            spatial_attention = block._spatial_attention
            hook = spatial_attention.register_forward_hook(hook_function)
            hooks.append(hook)

        with torch.no_grad():
            _ = model(sample_window, edge_index) 

        for hook in hooks:
            hook.remove()

        if attention_maps:
            att_matrix = np.mean(attention_maps[0], axis=0)  # [B, N, N] -> [N, N]
            
            plt.figure(figsize=(15, 12))
            plt.imshow(att_matrix, cmap='viridis', interpolation='nearest')
            plt.title('Spatial Attention Matrix')
            plt.colorbar()

            countries = [parameters.country_borders[i][1] for i in range(parameters.num_nodes)]
            plt.xticks(np.arange(parameters.num_nodes), countries, rotation=90)
            plt.yticks(np.arange(parameters.num_nodes), countries)

            plt.tight_layout()
            os.makedirs(f'{parameters.results_location}/{model_name}_{current_time}', exist_ok=True)
            plt.savefig(f'{parameters.results_location}/{model_name}_{current_time}/attention_matrix.png')
            plt.show()
        else:
            print("No attention weights captured.")
            
    except Exception as e:
        print(f"Error generating attention matrix: {str(e)}")

    plt.show()
    pkl.dump(model, open(f'{parameters.results_location}/{model_name}_{current_time}/model.pkl', 'wb'))
    