import os
from datetime import date, timedelta, datetime
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver
import time
from utils import *
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Download and aggregate weather data')
    parser.add_argument('-s', type=str, default='2023-01-01', 
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('-e', type=str, default='2023-12-31', 
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('-d', type=str, default='./temp', 
                        help='Folder to download temporary CSV files')
    parser.add_argument('--o', type=str, default='./countries', 
                        help='Folder to save aggregated country CSV files')
    parser.add_argument('--c', type=str, default='./cities', 
                        help='Folder to save individual city CSV files')
    
    return parser.parse_args()


def main():
    
    args = parse_arguments()
    
    os.makedirs(args.d, exist_ok=True)
    os.makedirs(args.o, exist_ok=True)
    os.makedirs(args.c, exist_ok=True)
    
    options = Options()
    prefs = {
        "download.default_directory": os.path.abspath(args.d),
        "download.prompt_for_download": False,
        "profile.default_content_settings.popups": 0,
        "profile.content_settings.exceptions.automatic_downloads.*.setting": 1
    }
    options.add_experimental_option("prefs", prefs)
    options.add_argument('--maximized')
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--window-size=1920,1080")    

    driver = WebDriver(service=Service(), options=options)

    driver.get('https://weatherdownloader.oikolab.com/app')
    elem = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, "bk-Column"))
            )
    
    # Convert start and end dates
    start_date = datetime.strptime(args.s, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.e, '%Y-%m-%d').date()
    
    for country, cities in cities_by_country.items():
        print(f"Downloading data for {country}")
        dfs = []
        for city in cities:
            
            select_city(driver, city)
            download_folder = args.d
            output_file = f'{args.c}/combined_data_{country}_{city}.csv'
            
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                
                if download_csv_for_date(driver, date_str, download_folder):
                    time.sleep(1)
                
                current_date += timedelta(days=1)
            
            concatenate_csvs(download_folder, output_file)
            dfs.append(pd.read_csv(output_file))

        mean_df = aggregate_city_dataframes(dfs)
        mean_df.to_csv(f'{args.o}/mean_data_{country}.csv')

    driver.quit()


if __name__ == "__main__":
    main()
