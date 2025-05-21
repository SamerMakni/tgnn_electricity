import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from torch_geometric_temporal.signal import temporal_signal_split
import pickle as pkl
import time
import json
from prepare_graph_signal import PyGDataset
import argparse
import parameters

time = time.strftime("%Y-%m-%d-%H-%M-%S")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline model')
    parser.add_argument('-l', '--length', type=int, default=500, help='Length of the snapshots')
    parser.add_argument('-n', '--normalize', type=str, choices=['all', 'features', 'target', 'None'], default='all',
                        help='Normalization type: all, features, target, or None')
    parser.add_argument('-s', '--standardizer', type=str, choices=['std', 'population'], default='population', help='Standardization type: std or population')
    args = parser.parse_args()
    
    temporal_dataset = PyGDataset(snapshot_length=args.length, normalize=args.normalize, normalizer=args.standardizer).temporal_dataset
    
    train_dataset, test_dataset = temporal_signal_split(temporal_dataset, train_ratio=0.8)
    train_features = np.array(train_dataset.features)  
    train_targets = np.array(train_dataset.targets)    
    test_features = np.array(test_dataset.features)   
    test_targets = np.array(test_dataset.targets)      

    num_nodes = train_targets.shape[1]
    train_time_steps = train_targets.shape[0]
    test_time_steps = test_targets.shape[0]

    train_X = train_features.reshape(train_time_steps, -1)  
    train_y = train_targets  
    test_X = test_features.reshape(test_time_steps, -1)  
    test_y = test_targets

    model = LinearRegression()
    model.fit(train_X, train_y)

    test_predictions = model.predict(test_X)

    mse = mean_squared_error(test_y, test_predictions)
    mae = mean_absolute_error(test_y, test_predictions)
    r2 = r2_score(test_y, test_predictions)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R² Score: {r2}')
    results_json = {
        'time': time,
        'model': 'Linear Regression',
        'snapshot_length': f'{args.length}',
        'normalize': f'{args.normalize}',
        'normalizer': f'{args.standardizer}',
        'mse': f'{mse}',
        'mae': f'{mae}',
        'r2': f'{r2}'
    }
        
    with open(f'{parameters.results_location}evaluations.json', 'r+') as f:
        data = json.load(f)
        data['baseline'].append(results_json)
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
        
    plt.figure(figsize=(8, 8))
    plt.scatter(test_y, test_predictions, alpha=0.5)
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    plt.savefig(f'{parameters.results_location}figures/baseline_scatter_{time}.png')
    plt.show()
                
    rows = int(np.ceil(np.sqrt(num_nodes)))
    cols = rows if rows * (rows - 1) < num_nodes else rows - 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()

    for node in range(num_nodes):
        ax = axes[node]
        
        ax.plot(range(test_time_steps), test_targets[:, node], label='Actual', color='blue')
        ax.plot(range(test_time_steps), test_predictions[:, node], label='Predicted', color='orange')
        
        country_name = parameters.country_borders[node][1]  # Get the country name
        ax.set_title(f'{country_name}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Target Signal')
        ax.legend()

    for i in range(num_nodes, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(f'{parameters.results_location}figures/baseline_predictions_{time}.png')
    plt.show()
