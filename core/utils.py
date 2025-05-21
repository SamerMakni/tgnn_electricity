import os
import shutil
import threading
import itertools
import sys
import time
from pathlib import Path
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
)

VALID_COUNTRIES = [
    'AL', 'AT', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 
    'GE', 'DE', 'GR', 'HU', 'IE', 'IT', 'XK', 'LV', 'LT', 'LU', 'ME', 'NL', 
    'MK', 'NO', 'PL', 'PT', 'RO', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'UA', 'UK'
]

COUNTRY_INFO = """
Supported country codes: 
AL (Albania), AT (Austria), BE (Belgium), BA (Bosnia and Herzegovina), BG (Bulgaria),
HR (Croatia), CY (Cyprus), CZ (Czech Republic), DK (Denmark), EE (Estonia), FI (Finland),
FR (France), GE (Georgia), DE (Germany), GR (Greece), HU (Hungary), IE (Ireland),
IT (Italy), XK (Kosovo), LV (Latvia), LT (Lithuania), LU (Luxembourg),
ME (Montenegro), NL (Netherlands), MK (North Macedonia), NO (Norway),
PL (Poland), PT (Portugal), RO (Romania), RS (Serbia), SK (Slovakia), SI (Slovenia),
ES (Spain), SE (Sweden), CH (Switzerland), UA (Ukraine), UK (United Kingdom)
"""

VALID_YEARS = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']

YEAR_INFO = "Supported years: %s-%s" % (VALID_YEARS[0], VALID_YEARS[-1])

URLS = {
    'login': 'https://transparency.entsoe.eu/protected-url',
    'total_load': 'https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show',
    'generator': 'https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show',
    'border': 'https://transparency.entsoe.eu/transmission-domain/r2/scheduledCommercialExchangesDayAhead/show'
}

PROGRESS_BAR = Progress(
    TextColumn("[bold white]{task.percentage:>3.0f}%"), 
    BarColumn(),
    MofNCompleteColumn()
)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Spinner:
    def __init__(self, message):
        self.message = message
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.spinner = itertools.cycle(self.steps)
        self.stop_running = False
        self.thread = threading.Thread(target=self.run)
    
    def start(self):
        self.stop_running = False
        sys.stdout.write(self.message)
        sys.stdout.flush()
        self.thread.start()

    def run(self):
        while not self.stop_running:
            sys.stdout.write(next(self.spinner))
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b')
            sys.stdout.flush()
    
    def stop(self):
        self.stop_running = True
        self.thread.join()
        sys.stdout.write('\b') 
        sys.stdout.flush() 
    
def select_file_name(new_filename, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    parent_dir = str(Path(download_dir).parent.absolute())
    print(f"{bcolors.OKCYAN}  Current files in {parent_dir}:")
    current_filename = max([parent_dir + "/" + f for f in os.listdir(parent_dir)],key=os.path.getctime)
    if current_filename.endswith('.csv'):
        saved_filename = os.path.join(download_dir,new_filename)
        shutil.move(current_filename, saved_filename)
        print(f"{bcolors.OKGREEN}  File saved as: {saved_filename}")
    else:
        print(f"{bcolors.FAIL}  No file to save.")
        
        

