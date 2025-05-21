import pandas as pd
import numpy as np
import glob
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, DynamicGraphTemporalSignal
import pickle as pkl
import sys
sys.path.append('../')
from core.utils import bcolors, PROGRESS_BAR
sys.path.append('.')
import tqdm
import warnings
warnings.filterwarnings("ignore")
import parameters
from rich.console import Console
from rich.markdown import Markdown

from scipy.stats import pearsonr
from math import sqrt


def compute_euclidean_edge_weights(country_borders, num_snapshots=8760):
    # Extract edges from country_borders
    edges = []
    for node, (neighbors, _, _, _, _) in country_borders.items():
        for neighbor in neighbors:
            if node < neighbor:  # Avoid duplicates in undirected graph
                edges.append((node, neighbor))
    
    num_edges = len(edges)
    edge_index = np.array(edges).T  # Shape: (2, num_edges)
    
    # Compute Euclidean distances for each edge using coordinates from country_borders
    distances = []
    for i, j in zip(edge_index[0], edge_index[1]):
        lat1, lon1 = country_borders[i][4]  # Fifth element is (lat, lon
        lat2, lon2 = country_borders[j][4]
        dist = sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)  # Euclidean distance
        distances.append(dist)

    # Create edge weights array: duplicate distances across all snapshots
    edge_weights = np.tile(distances, (num_snapshots, 1))  # Shape: (num_snapshots, num_edges)
    
    return edge_weights


def compute_edge_weights(country_borders, energy_prices, num_snapshots=8760):
    edges = []
    for node, (neighbors, _, _, _) in country_borders.items():
        for neighbor in neighbors:
            if node < neighbor:  # Avoid duplicates in undirected graph
                edges.append((node, neighbor))
    
    num_edges = len(edges)
    edge_index = np.array(edges).T  # Shape: (2, num_edges)

    # Compute correlations for each edge
    correlations = []
    for i, j in zip(edge_index[0], edge_index[1]):
        price_i = energy_prices[:, i]
        price_j = energy_prices[:, j]
        corr, _ = pearsonr(price_i, price_j)
        correlations.append(corr)

    # Create edge weights array: duplicate correlations across all snapshots
    edge_weights = np.tile(correlations, (num_snapshots, 1))  # Shape: (num_snapshots, num_edges)
    
    return edge_weights

class PyGDataset:
    def __init__(self, data_location='.',
                 normalize='target', selected_countries='all', snapshot_length=500,
                 connections='borders', export_location='./temporal_dataset.pkl',
                 normalizer='population', date_features=False, weather_features=False, generator_features=False ,edge_weights=None, target='ahead', task='price', delta=24, lags=6):
        
        self.data_location = data_location
        self.selected_countries = selected_countries
        self.snapshot_length = snapshot_length
        self.connections = connections
        self.export_location = export_location
        self.normalizer = normalizer
        self.date_features = date_features
        self.weather_features = weather_features
        self.generator_features = generator_features
        self.country_borders = parameters.country_borders
        self.target = target
        self.lags = lags
        self.delta = delta
        self.edge_weights = edge_weights
        self.task = task
        self.normalize = None if self.weather_features and self.date_features and self.generator_features else normalize

        
        self.temporal_dataset, self.edge_index, self.skipped_countries = self.create_dataset()
    
    def create_edge_index(self):
        """Creates an edge index based on the country borders, ensuring contiguous indices."""
        edge_list = []
        
        # Select the countries to include
        selected_countries = self.selected_countries if self.selected_countries != 'all' else list(self.country_borders.keys())
        
        # Create a mapping from original indices to new contiguous indices
        index_map = {country: i for i, country in enumerate(selected_countries)}

        for country, (neighbors, _, _, _, _) in self.country_borders.items():
            if country in selected_countries:
                for neighbor in neighbors:
                    if neighbor in selected_countries:
                        edge_list.append([index_map[country], index_map[neighbor]])
        
        edge_index = np.array(edge_list).T
        return edge_index
    
    def create_edge_weights(self):
        """Creates edge weights ensuring contiguous indices even with missing countries."""
        df = pd.read_csv(f'{self.data_location}all_net_flows_2023.csv')
        
        # Get edge index and ensure contiguous remapping
        selected_countries = self.selected_countries if self.selected_countries != 'all' else list(self.country_borders.keys())
        index_map = {country: i for i, country in enumerate(selected_countries)}
        edge_index = self.create_edge_index()

        edge_weights_sequence = []
        
        for t in range(len(df)):  # Iterate over all timestamps
            edge_weights = []
            for src, tgt in zip(edge_index[0], edge_index[1]):
                # Retrieve original country indices before remapping
                original_src = selected_countries[src]
                original_tgt = selected_countries[tgt]

                # Use country codes (ISO codes) for column lookup
                column_name = f"{self.country_borders[original_src][3]}_{self.country_borders[original_tgt][3]}"
                reversed_column_name = f"{self.country_borders[original_tgt][3]}_{self.country_borders[original_src][3]}"

                # Assign edge weights based on the correct columns
                if column_name in df.columns:
                    edge_weights.append(df.at[t, column_name])
                else:
                    edge_weights.append(df.at[t, reversed_column_name] if reversed_column_name in df.columns else 0)

            edge_weights_sequence.append(np.array(edge_weights))

        return np.array(edge_weights_sequence)

        

    def normalize_by_population(self, df, country_name):
        """Normalizes the dataframe by population of the country."""
        for item in self.country_borders:
            if self.country_borders[item][1] == country_name:
                country_name = item
                break
        population = self.country_borders[country_name][2]
        df = df.apply(lambda x: x / population)
        return df

    def preprocess_country_dataframes(self, dataframes):
        """Preprocesses the country dataframes by normalizing and handling missing values."""
        node_features_list = []
        target_list = []
        country_names = []
        skipped_countries = []
        loop = [c for c in dataframes if self.selected_countries == 'all' or c['Area'].iloc[0] in self.selected_countries]

        with PROGRESS_BAR as p:
            Console().print(Markdown("""# Preprocessing dataframes"""))
            for country_df in p.track(loop):
                df_copy = country_df.copy()
                if self.snapshot_length != 'all':
                    df_copy = df_copy.head(self.snapshot_length)

                if 'Area' in df_copy.columns:
                    df_copy.fillna(df_copy.mean(), inplace=True)
                    country_name = df_copy['Area'].unique()[0]
                    country_name = self.country_borders[country_name][1]
                    df_copy.drop(columns=['Area'], inplace=True)
                else:
                    raise ValueError("No 'Area' column found in the dataframe")    
                
                date_columns = ['day sin', 'day cos', 'hour sin', 'hour cos', 'weekday sin', 'weekday cos']
                weather_columns = ['dewpoint_temperature (degC)','relative_humidity (0-1)','temperature (degC)','total_cloud_cover (0-1)','total_precipitation (mm of water equivalent)','wind_direction (deg)','wind_speed (m/s)']
                generator_features = ['Unnamed: 0','Biomass  - Actual Aggregated [MW]','Energy storage - Actual Aggregated [MW]','Fossil Brown coal/Lignite  - Actual Aggregated [MW]','Fossil Coal-derived gas  - Actual Aggregated [MW]','Fossil Gas  - Actual Aggregated [MW]','Fossil Hard coal  - Actual Aggregated [MW]','Fossil Oil  - Actual Aggregated [MW]','Fossil Oil shale  - Actual Aggregated [MW]','Fossil Peat  - Actual Aggregated [MW]','Geothermal  - Actual Aggregated [MW]','Hydro Pumped Storage  - Actual Aggregated [MW]','Hydro Pumped Storage  - Actual Consumption [MW]','Hydro Run-of-river and poundage  - Actual Aggregated [MW]','Hydro Water Reservoir  - Actual Aggregated [MW]','Marine  - Actual Aggregated [MW]','Nuclear  - Actual Aggregated [MW]','Other  - Actual Aggregated [MW]','Other renewable  - Actual Aggregated [MW]','Solar  - Actual Aggregated [MW]','Waste  - Actual Aggregated [MW]','Wind Offshore  - Actual Aggregated [MW]','Wind Onshore  - Actual Aggregated [MW]']
                if not self.date_features:
                    df_copy = df_copy.drop(columns=date_columns)
                    
                if not self.weather_features:
                    df_copy = df_copy.drop(columns=weather_columns)
                
                if not self.generator_features:
                    df_copy = df_copy.drop(columns=generator_features)
                    
                df_copy = df_copy.apply(pd.to_numeric, errors='coerce').fillna(df_copy.mean())
                df_copy = df_copy.replace(0, df_copy.mean())

                if self.normalize == 'all':
                    columns_to_normalize = [col for col in df_copy.columns if col not in date_columns]
                elif self.normalize == 'target':
                    columns_to_normalize = ['Actual Total Load [MW]']
                elif self.normalize == 'features':
                    columns_to_normalize = [col for col in df_copy.columns if col not in date_columns and col != 'Actual Total Load [MW]']

                if self.normalize:
                    if self.normalizer == 'std':
                        df_copy[columns_to_normalize] = (df_copy[columns_to_normalize] - df_copy[columns_to_normalize].mean()) / df_copy[columns_to_normalize].std()
                    elif self.normalizer == 'population':
                        df_copy[columns_to_normalize] = df_copy[columns_to_normalize].apply(lambda x: self.normalize_by_population(x, country_name))

                if self.task == 'price':
                    target_column = 'Price (EUR/MWhe)'
                    targets = df_copy[target_column].shift(-self.delta).fillna(method='ffill').values  # Shift for next delta
                    columns = df_copy.columns.tolist()
                    # print(f"Columns: {columns}")
                    for i in range(1, self.lags + 1):
                        df_copy[f'{target_column} -{i}'] = df_copy[target_column].shift(i)
    
                
                    node_features = df_copy.drop(columns=['Day-ahead Total Load Forecast [MW]', 'Actual Total Load [MW]', 'MTU', 'Datetime (UTC)']).values
                    
                if self.task == 'load':
                    target_column = 'Actual Total Load [MW]'
                    df_copy = df_copy.drop(columns=['Price (EUR/MWhe)'])
                    columns = df_copy.columns.tolist()
                    # print(f"Columns: {columns}")
                    for i in range(1, self.lags + 1):
                        df_copy[f'{target_column} -{i}'] = df_copy[target_column].shift(i)
    
                    if self.target == 'current':
                        targets = df_copy[target_column].values
                        node_features = df_copy.drop(columns=['Day-ahead Total Load Forecast [MW]', 'Actual Total Load [MW]', 'MTU', 'Datetime (UTC)']).values
                    elif self.target == 'ahead':
                        targets = df_copy[target_column].shift(-self.delta).fillna(method='ffill').values  # Shift for next delta
                        node_features = df_copy.drop(columns=['Day-ahead Total Load Forecast [MW]', 'MTU', 'Datetime (UTC)']).values


                if df_copy[target_column].isna().all():
                    skipped_countries.append(country_name)

                country_names.append(country_name)
                node_features_list.append(node_features)
                target_list.append(targets)

        all_node_features = np.stack(node_features_list, axis=0)
        all_targets = np.stack(target_list, axis=0)

        return all_node_features, all_targets, skipped_countries


    def create_temporal_signal_from_dataframes(self, dataframes):
        """Creates a StaticGraphTemporalSignal from dataframes."""
        node_features, targets, skipped_countries = self.preprocess_country_dataframes(dataframes)
        
        num_nodes = len(node_features)
        num_time_steps = node_features.shape[1]

        if self.connections == 'fully':
            edge_index = np.array([[i, j] for i in range(num_nodes) for j in range(num_nodes)]).T
        else:
            edge_index = self.create_edge_index()

        node_features_list = [np.array(node_features[:, i, :], dtype=np.float32) for i in range(num_time_steps)]
        targets_array = np.array(targets.T, dtype=np.float32)

        for i in range(len(node_features_list)):
            node_features_list[i] = np.nan_to_num(node_features_list[i], nan=np.nanmean(node_features_list[i]))

        targets_array = np.nan_to_num(targets_array, nan=np.nanmean(targets_array))
        
        if self.edge_weights:
            
            if self.edge_weights == 'load':
                dataset = DynamicGraphTemporalSignal(
                    edge_indices=[edge_index] * self.snapshot_length,
                    edge_weights=self.create_edge_weights(),
                    features=node_features_list,
                    targets=list(targets_array)
                )   
            if self.edge_weights == 'price':
                targets = np.vstack(targets)
                w = compute_edge_weights(self.country_borders, targets, self.snapshot_length)
                print(w)
                dataset = DynamicGraphTemporalSignal(
                    edge_indices=[edge_index] * self.snapshot_length,
                    edge_weights=w,
                    features=node_features_list,
                    targets=list(targets_array)
                )
            if self.edge_weights == "euclidean":
                w = compute_euclidean_edge_weights(self.country_borders, self.snapshot_length)
                dataset = DynamicGraphTemporalSignal(
                    edge_indices=[edge_index] * self.snapshot_length,
                    edge_weights=w,
                    features=node_features_list,
                    targets=list(targets_array)
                )
        else:
            dataset = StaticGraphTemporalSignal(
                edge_index=edge_index,
                edge_weight=np.ones(edge_index.shape[1]),  
                features=node_features_list,
                targets=list(targets_array)
            )
            
        return dataset, edge_index, skipped_countries

    def create_dataset(self):
        """Generates the temporal dataset based on the dataframes."""
        data_frames = []
        dfs = glob.glob(f'{self.data_location}/*_*_data.csv')
        dfs.sort()
        for df in dfs:
            df = pd.read_csv(df)
            df_area = df['Area'].unique()[0]
            df_code = [k for k, v in self.country_borders.items() if v[1] == df_area]
            if not df_code:
                continue
            df['Area'] = df_code[0]
            data_frames.append(df)

        temporal_dataset, edge_index, skipped_countries = self.create_temporal_signal_from_dataframes(data_frames)
        
        if skipped_countries:
            Console().print(f'[bold dark_orange]:warning:[/bold dark_orange] [dark_orange]Null Countries (all targets for selected period are 0 or NaN):[/dark_orange] © [white]{skipped_countries}')
        
        return temporal_dataset, edge_index, skipped_countries

    def describe(self):
        """Describes the temporal dataset."""
        num_nodes = self.temporal_dataset.features[0].shape[0]
        num_features_per_node = self.temporal_dataset.features[0].shape[1]
        num_time_steps = len(self.temporal_dataset.features)

        print(f"Number of nodes: {num_nodes}")
        print(f"Number of features per node: {num_features_per_node}")
        print(f"Number of snapshots: {num_time_steps}")
        print(f"Target shape: {len(self.temporal_dataset.targets)} x {self.temporal_dataset.targets[0].shape[0]}")

    def export(self):
        """Exports the temporal dataset to a pickle file."""
        with open(self.export_location, 'wb') as f:
            pkl.dump(self.temporal_dataset, f)
        print(f"{bcolors.OKGREEN}Temporal dataset created and saved successfully at {self.export_location}{bcolors.ENDC}")
