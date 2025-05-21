from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import argparse
import parameters
import json

from torch_geometric_temporal import temporal_signal_split
from torch_geometric.data import DataLoader

from models import GConvReccurent, GConvAttention, STConv, GConvLSTM
from prepare_graph_signal import PyGDataset

from rich.console import Console
from rich.markdown import Markdown


def main(args):
    # Load the dataset
    temporal_dataset = PyGDataset(data_location=parameters.data_location,
                                   normalize=parameters.normalize,
                                   selected_countries=parameters.selected_countries,
                                   snapshot_length=parameters.snapshot_length,
                                   connections=parameters.connections,
                                   export_location=parameters.export_location,
                                   normalizer=parameters.normalizer,
                                   weather_features=parameters.weather_features,
                                   generator_features=parameters.generator_features,  
                                   date_features=parameters.date_features,                                   
                                   edge_weights=parameters.edge_weights,
                                   lags=parameters.lags,
                                   ).temporal_dataset

    train_dataset, test_dataset = temporal_signal_split(temporal_dataset, train_ratio=parameters.ratio)
    
    train_batch = DataLoader(list(train_dataset), batch_size=parameters.batch_size, shuffle=True)
    test_batch = DataLoader(list(test_dataset), batch_size=parameters.batch_size, shuffle=False)
    
    node_features = train_dataset[0].x.shape[1]
    nodes_number = train_dataset[0].x.shape[0]
    print(nodes_number)
    

    experiment = None
    if not args.debug:
        experiment = Experiment(api_key="",
                                project_name="",
                                workspace="")
        if args.name:
            experiment.set_name(args.name)

    for m in parameters.models:
        if m[0] == 'GConvLSTM':
            model = GConvLSTM.RecurrentGCN(node_features=node_features,window_length=parameters.window, device=parameters.device).to(parameters.device)
            parameters_count = model.count_parameters()
            
            if experiment:
                hyper_params = {
                    'model': m[0],
                    'hidden_size': m[2],
                    'lr': parameters.lr,
                    'parameters': parameters_count
                }
                experiment.log_parameters(hyper_params)

            Console().print(Markdown(f'# Training {m[0]} (*{m[2]} hidden size*, *{parameters.lr} learning rate*, *{parameters_count} parameters*)'))
            Console().print(Markdown(f'Selected device: *{parameters.device}*'))
            model, loss_history, elapsed_time = GConvLSTM.train(model, train_dataset)
            mse, mae, r2 = GConvLSTM.eval(model, test_dataset)
            Console().print(Markdown(f'**Test dataset**'))
            Console().print(Markdown(f'**MSE**: *{mse:0.3f}* | **MAE**: *{mae:0.3f}* | **R2**: *{r2:0.3f}*'))
            GConvLSTM.report(model, test_dataset, loss_history, m[0])
            Console().print(f'[bold white] Model and Performance plots exported to [/bold white] {parameters.results_location}')

            if experiment:
                experiment.log_metrics({'MSE': mse, 'MAE': mae, 'R2': r2, 'elapsed_time': elapsed_time})
                experiment.set_model_graph(model)
                log_model(experiment, model=model, model_name=m[0])


        if m[0] == 'STConv' or m[0] == 'ASTGCN':
            if m[0] == 'STConv':
                model = STConv.AttentionGCN(hidden_size=m[2], num_features=node_features, num_nodes=nodes_number).to(parameters.device)
            else:
                model = GConvAttention.AttentionGCN(cell=None, hidden_size=m[2], node_features=node_features, num_nodes=nodes_number).to(parameters.device)
            parameters_count = model.count_parameters()
            Console().print(Markdown(f'# Training {m[0]} (*{m[2]} hidden size*, *{parameters.lr} learning rate*, *{parameters_count} parameters*)'))
            Console().print(Markdown(f'Selected device: *{parameters.device}*'))
            if experiment:
                hyper_params = {
                    'model': m[0],
                    'hidden_size': m[2],
                    'lr': parameters.lr,
                    'parameters': parameters_count
                }
                experiment.log_parameters(hyper_params)
                
            if m[0] == 'STConv':
                model, loss_history, elapsed_time = STConv.train(model, train_batch)
                mse, mae, r2 = STConv.eval(model, test_batch)
                Console().print(Markdown(f'**Test dataset**'))
                Console().print(Markdown(f'**MSE**: *{mse:0.3f}* | **MAE**: *{mae:0.3f}* | **R2**: *{r2:0.3f}*'))
                STConv.report(model, test_batch, loss_history, m[0])
                Console().print(f'[bold white] Model and Performance plots exported to [/bold white] {parameters.results_location}')

                if experiment:
                    experiment.log_metrics({'MSE': mse, 'MAE': mae, 'R2': r2, 'elapsed_time': elapsed_time})
                    experiment.set_model_graph(model)
                    log_model(experiment, model=model, model_name=m[0])
            else:
                    
                model, loss_history, elapsed_time = GConvAttention.train(model, train_batch, test_batch)
                mse, mae, r2 = GConvAttention.eval(model, test_batch)
                Console().print(Markdown(f'**Test dataset**'))
                Console().print(Markdown(f'**MSE**: *{mse:0.3f}* | **MAE**: *{mae:0.3f}* | **R2**: *{r2:0.3f}*'))
                GConvAttention.report(model, test_batch, loss_history, m[0])
                Console().print(f'[bold white] Model and Performance plots exported to [/bold white] {parameters.results_location}')

                if experiment:
                    experiment.log_metrics({'MSE': mse, 'MAE': mae, 'R2': r2, 'elapsed_time': elapsed_time})
                    experiment.set_model_graph(model)
                    log_model(experiment, model=model, model_name=m[0])
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Graph Neural Network Models.')
    parser.add_argument('--name', type=str, help='Name for the Comet ML experiment')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to skip Comet ML logging')
    
    args = parser.parse_args()
    main(args)
