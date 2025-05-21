from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU
from cuda_selector import auto_cuda
import pandas as pd


def filter_countries(country_borders, selected_indices):
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
    
    new_country_borders = {}
    for old_idx, new_idx in index_mapping.items():  # Fix here
        neighbors, name, population, code, coords = country_borders[old_idx]
        new_neighbors = [index_mapping[n] for n in neighbors if n in index_mapping]
        new_country_borders[new_idx] = (new_neighbors, name, population, code, coords)
    
    return new_country_borders

# General parameters
results_location = '' # should be changed 
data_location = '' # location of the aggregated data that were created using the aggregation script


# Parameters for training
models = [
    # ('GConvLSTM', GConvLSTM, 256),
    #('GConvGRU', GConvGRU, 256)
    ('ASTGCN', None, 128)
    # ('STConv', None, 128)
]

batch_size = 128
epochs = 450
stride = 1
device = auto_cuda()
lr = 0.001
ratio = 0.8
window = 10 # this is not being used now

# Parameters for dataset
target = 'ahead' # or 'current'
task = 'price' # or 'load'
delta = 24
date_features = True
weather_features = False
generator_features = False

lags = 0
num_features = pd.read_csv(f'{data_location}AT_2023_data.csv').shape[1] - 2
snapshot_length = 8760 # 24*7*32 # max is 8760
normalizer = None # or 'std'
normalize = None # or 'None' or 'features' or 'target'
selected_countries =  'all'# or specifiy country code ()
connections = 'borders' # or 'fully'
export_location = '../' # location to export the dataset
edge_weights = "euclidean" # or 'price' or 'euclidean' or 'none'
# countries dict -> code: ([borders], name, population)



countries_dict = {
    0: ([23, 24, 36, 15], 'Albania (AL)', 2.778, 'AL', (41.1533, 20.1683)), 
    1: ([8, 7, 33, 17, 32, 19, 5], 'Austria (AT)', 8.859, 'AT', (47.5162, 14.5501)),
    2: ([16, 30, 23], 'Bosnia and Herz. (BA)', 3.301, 'BA', (43.9159, 17.6791)),
    3: ([13, 21, 25, 8], 'Belgium (BE)', 11.46, 'BE', (50.8333, 4.0)),
    4: ([29, 30, 24, 15], 'Bulgaria (BG)', 7.101, 'BG', (42.7339, 25.4858)),
    5: ([13, 1, 8, 19], 'Switzerland (CH)', 8.57, 'CH', (46.8182, 8.2275)),
    6: ([], 'Cyprus (CY)', 1.198, 'CY', (35.1667, 33.3667)),
    7: ([8, 1, 33, 27], 'Czech Republic (CZ)', 10.65, 'CZ', (49.75, 15.5)),
    8: ([13, 1, 7, 3, 5, 21, 25, 9], 'Germany (DE)', 83.02, 'DE', (51.0, 9.0)),
    9: ([8, 31], 'Denmark (DK)', 5.806, 'DK', (56.0, 10.0)),
    10: ([22], 'Estonia (EE)', 1.328, 'EE', (59.0, 26.0)),
    11: ([13, 28], 'Spain (ES)', 46.94, 'ES', (40.0, -4.0)),
    12: ([26, 31], 'Finland (FI)', 5.518, 'FI', (64.0, 26.0)),
    13: ([3, 21, 8, 5, 19, 11], 'France (FR)', 67.06, 'FR', (46.0, 2.0)),
    14: ([], 'Georgia (GE)', 3.717, 'GE' , (42.0, 43.5)),
    15: ([0, 4, 25], 'Greece (GR)', 10.72, 'GR', (39.0742, 21.8243)),
    16: ([32, 23, 30, 2, 17], 'Croatia (HR)', 4.105, 'HR', (45.1667, 15.5)),
    17: ([16, 33, 32, 1, 34, 29], 'Hungary (HU)', 9.772, 'HU', (47.0, 20.)),
    18: ([35], 'Ireland (IE)', 4.857, 'IE', (53.0, -8.0)),
    19: ([13, 1, 32, 5], 'Italy (IT)', 60.36, 'IT', (42.8333, 12.8333)),
    20: ([22, 27], 'Lithuania (LT)', 2.794, 'LT', (56.0, 24.0)),
    21: ([3, 13, 8], 'Luxembourg (LU)', 0.626, 'LU', (49.75, 6.1667)),
    22: ([20, 10], 'Latvia (LV)', 1.919, 'LV', (57.0, 25.)),
    23: ([36, 0, 30, 2, 16], 'Montenegro (ME)', 0.622, 'ME', (42.5, 19.3)),
    24: ([36, 30, 4, 15, 0], 'North Macedonia (MK)', 2.083, 'MK', (41.6086, 21.7453)),
    25: ([3, 8], 'Netherlands (NL)', 17.28, 'NL', (52.5, 5.75)),
    26: ([31, 12], 'Norway (NO)', 5.368, 'NO', (62.0, 10.0)),
    27: ([8, 7, 33, 34, 20], 'Poland (PL)', 38.43, 'PL', (52.0, 20.0)),
    28: ([11], 'Portugal (PT)', 10.29, 'PT', (39.5, -8.0)),
    29: ([34, 4, 30, 17], 'Romania (RO)', 19.41, 'RO', (46.0, 25.0)),
    30: ([17, 29, 4, 24, 16, 2, 22], 'Serbia (RS)', 7.001, 'RS', (44.0, 21.0)),
    31: ([26, 12], 'Sweden (SE)', 10.23, 'SE', (62.0, 15.0)),
    32: ([1, 16, 17, 19], 'Slovenia (SI)', 2.07, 'SI', (46.0, 15.0)),
    33: ([7, 1, 17, 34, 27], 'Slovakia (SK)', 5.457, 'SK', (48.6667, 19.5)),
    34: ([27, 33, 17, 29], 'Ukraine (UA)', 41.98, 'UA', (49.0, 32.0)),
    35: ([16], 'United Kingdom (UK)' , 66.65, 'UK', (54.0, -2.0)),
    36: ([23, 30, 0, 24], 'Kosovo (XK)', 1.798, 'XK', (42.5617, 20.3400))
}
selected = [1, 3, 5, 7, 8, 9, 11, 17, 25, 27, 13]
country_borders = filter_countries(countries_dict, selected)

num_nodes = len(country_borders)