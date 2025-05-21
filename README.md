This project currently has two major parts:

- The first part is a scraper that downloads data from the ENTSO-E Transparency Platform. A Telegram bot is also available to start and monitor the scraper.
- The second part is related to modeling anf forecasting where we deploy dynamic graph neural networks to forecast the total load for various countries.

# A.) Entsoe Downloader

## A.1. Dependencies

First, create a Python 3.10 virtual environment with the following command:
```bash
conda create -n entsoe_scraper python=3.10
```
Then, activate the virtual environment before installing the required dependencies:
```bash
conda activate entsoe_scraper
```

Next, you need to install the required dependencies for the scraper and the telegram bot. You can do this by running the following commands in the root directory of the project:
```bash
pip install -r requirements_scraper.txt
pip install -r requirements_general.txt
```

## A.2. Usage

### 1. Entsoe Scraper

#### Running from server 

- Use the `--headless` flag to run the scraper in headless mode, which allow this script to run on a server.
- If you are not using the `--headless` flag, you have to download Chrome drivers from [here](https://googlechromelabs.github.io/chrome-for-testing/).

#### Download Options
You can specify the type of data to download using the `--download` flag. The available options are:
- `total_load`: to download total load data
- `generator`: to download generation per Production Type data
- `border`: to download border transmission data
- `all`: to download all the data

#### Additional Options
- `--headless`: Run the browser in headless mode.
- `--login_email`: Specify the email for logging in. Default is set in the `config.py` file.
- `--login_password`: Specify the password for logging in. Default is set in the `config.py` file.
- `--data_dir`: Set the path to save downloaded files. Default is `./downloads`, preferably changed to somewhere in **idms**.
- `--countries`: Specify countries to download data for. Acceptable values are:
  - `AL`, `AT`, `BE`, `BA`, `BG`, `HR`, `CY`, `CZ`, `DK`, `EE`, `FI`, `FR`, 
    `GE`, `DE`, `GR`, `HU`, `IE`, `IT`, `XK`, `LV`, `LT`, `LU`, `ME`, `NL`, 
    `MK`, `NO`, `PL`, `PT`, `RO`, `RS`, `SK`, `SI`, `ES`, `SE`, `CH`, `UA`, `UK`.
- `--years`: Specify years for which to download data. Acceptable values are:
  - `2014`, `2015`, `2016`, `2017`, `2018`, `2019`, `2020`, `2021`, `2022`, `2023`, `2024`.

#### Example Command
For example, to download generator data for Hungary and Slovakia in headless mode only for the year 2024:
```bash
python3 scraper.py --headless --download generator --countries HU SK --years 2024
```

#### Additional Configuration
The default login credentials and script speed, can be modified in the `config.py` file.

### 2. Weather Scraper

### Running from server

- The weather scraper scrapes hourly data from OIKOLA API. 
- For unclear reasons (probably due non existent chrome-driver), this script does not work on GPU100, however it works on HULK2 and GPU1.
- The script is run in headless mode by default.

#### Options
- `-s`: Start date in the format `YYYY-MM-DD`.
- `-e`: End date in the format `YYYY-MM-DD`.
- `-d`: Directory to save the temp files.
- `-c`: Folder to save individual city CSV files.
- `-o`: Directory for output; aggregated country CSV files.
-
#### Example Command
```bash
python3 weather.py -s 2023-01-01 -e 2023-02-01  -o .
```

## A.3. Telegram Bot

A telegram bot is available to this repository that can ease data downloading. Basically, you can start and monitor the scraper from your telegram account.

### Setup

1. Create a new bot using the BotFather on Telegram. You can find the instructions [here](https://core.telegram.org/bots#6-botfather)

2. Copy the token provided by the BotFather and paste it in the `config.json` file. PLEASE always make your own configuration file.

For example include the following section in the `config.json` file:
```json
"telegram": {
    "token": "YOUR_TOKEN"
}
```

3. Also you have to specify your 'OPENAI_API_KEY' as the bot can use the OpenAI API to generate messages. Please create an `.env` file beside the `bot.py` file and include the following line:
```bash
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

4. Start the bot by running the following command:
```bash
python3 telegram_bot.py PATH_TO_CONFIG_JSON PATH_TO_DATA_DIR
```

Where `PATH_TO_CONFIG_JSON` is the path to the `config.json` file and `PATH_TO_DATA_DIR` is the path to the directory where the downloaded files are saved.

# B. Modeling and Forecasting

## B.1 Dependencies

We recommend to create a separate conda environment for the modeling part as dealing with PyTorch and GPU support can be tricky. You can create a new environment with the following command:
```bash
conda create -n entsoe_modeling python=3.10
```
Then, activate the virtual environment before installing the required dependencies:
```bash
conda activate entsoe_modeling
```

Next, you need to install the required dependencies for the modeling part. You can do this by running the following commands in the root directory of the project:
```bash
pip install -r requirements_modeling.txt
pip install -r requirements_general.txt
```

## B.2 Usage

The resources for modeling are available in the `training` directory.

```bash
cd training
```

### Data Preparation

You can use the import and use the class `PyGDataset()` class from the `prepare_graph_signal.py` file to prepare the data for the modeling part. The class initializes with the following parameters:
- `data_location`: Path to the directory where the aggregated data is stored.
- `normalize`: `all` to normalize all the data, `features` to normalize only the features, `target` to normalize only the target, and `None` to not normalize the data.
- `selected_countries`: List of two letter countries codes to include.
- `snapshot_length`: Number of hours to include, max is 8760.
- `connections`: `borders` or `fully` (borders to be fixed)
- `export_location`: Path to the directory where the prepared data will be pickled.
- `normalizer`: `std` or `population`
- `date_features`: `True` to include date features, `False` to exclude them.

### Training

For training there is no need to use the `PyGDataset()` class, you can just run `run_experiments.py` file which takes parameters for both data perparation and training from the file `parameters.py`. You can modify the parameters in the `parameters.py`. You can use the arguments `--name` to specify the name of the experiment that will be exported to comet-ml. You can also use the `--debug` flag to run the experiments without exporting to comet-ml.

```bash
python3 run_experiments.py --debug
```
To run the experiments with comet-ml:

```bash
python3 run_experiments.py --name "Experiment Name"
```
