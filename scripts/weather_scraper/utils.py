import os
from datetime import date, timedelta
import pandas as pd

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver
import time, os
import traceback
from selenium.webdriver.common.keys import Keys


cities_by_country = {
    "Austria": ["Vienna", "Salzburg"],
    "Belgium": ["Brussels", "Antwerp"],
    "Bulgaria": ["Sofia", "Varna"],
    "Croatia": ["Zagreb", "Dubrovnik"],
    "Cyprus": ["Nicosia", "Paphos"],
    "Czech Republic": ["Prague", "Ostrava"],
   "Denmark": ["Copenhagen", "Aalborg"],
   "Estonia": ["Tallinn", "Narva"],
  "Finland": ["Helsinki", "Oulu"],
   "France": ["Paris", "Marseille", "Lille"],
   "Germany": ["Berlin", "Munich", "Hamburg"],
   "Greece": ["Athens", "Thessaloniki"],
  "Hungary": ["Budapest", "Debrecen"],
   "Ireland": ["Dublin", "Cork"],
   "Italy": ["Rome", "Milan"],
   "Latvia": ["Riga", "Daugavpils"],
   "Lithuania": ["Vilnius", "Klaipėda"],
   "Luxembourg": ["Luxembourg City", "Esch-sur-Alzette"],
   "Malta": ["Valletta", "Qormi"],
   "Netherlands": ["Amsterdam", "Eindhoven"],
   "Poland": ["Warsaw", "Kraków"],
   "Portugal": ["Lisbon", "Porto"],
   "Romania": ["Bucharest", "Cluj-Napoca"],
   "Slovakia": ["Bratislava", "Košice"],
   "Slovenia": ["Ljubljana", "Maribor"],
   "Spain": ["Madrid", "Barcelona"],
   "Sweden": ["Stockholm", "Malmö"],
   "Switzerland": ["Zurich", "Geneva"],
   "Norway": ["Oslo", "Tromsø"],
   "United Kingdom": ["London", "Glasgow"]
}

def expand_shadow_element(driver, element):
    shadow_root = driver.execute_script('return arguments[0].shadowRoot', element)
    return shadow_root

def concatenate_csvs(download_folder, output_file):
    csv_files = [f for f in os.listdir(download_folder) if f.endswith('.csv')]
    
    combined_df = pd.concat([pd.read_csv(os.path.join(download_folder, f)) for f in csv_files])
    
    combined_df.to_csv(output_file, index=False)
    
    for f in csv_files:
        os.remove(os.path.join(download_folder, f))
        
        
def file_count(download_folder):
    return len([f for f in os.listdir(download_folder) if f.endswith('.csv')])


def rename_latest_file(download_folder, name):
    files = os.listdir(download_folder)
    files.sort(key=lambda x: os.path.getctime(os.path.join(download_folder, x)))
    latest_file = files[-1]
    new_name = os.path.join(download_folder, name)
    os.rename(os.path.join(download_folder, latest_file), new_name)
    return new_name
    
        
def download_csv_for_date(driver, new_date, download_folder):
    
    try:
        main = driver.find_element(By.CLASS_NAME, "bk-Column")
        shadow = expand_shadow_element(driver, main)
        
        all_divs = shadow.find_elements(By.CSS_SELECTOR, "div")
        nested_shadow = expand_shadow_element(driver, all_divs[0])
        # print(nested_shadow)
        
        div = nested_shadow.find_elements(By.CSS_SELECTOR, "div")[2]
        shadow = expand_shadow_element(driver, div)
        div = shadow.find_elements(By.CSS_SELECTOR, "div")[1]
        shadow = expand_shadow_element(driver, div)
        div = shadow.find_elements(By.CSS_SELECTOR, "div")[0]
        shadow = expand_shadow_element(driver, div)
        div = shadow.find_elements(By.CSS_SELECTOR, "div")[1]
        shadow = expand_shadow_element(driver, div)
        div = shadow.find_elements(By.CSS_SELECTOR, "div")[0]
        shadow = expand_shadow_element(driver, div)
        div = shadow.find_elements(By.CSS_SELECTOR, "div")[0]
        shadow = expand_shadow_element(driver, div)
        # date input 
        date_input = shadow.find_elements(By.CSS_SELECTOR, "input")[0]
        
        driver.execute_script("""
            let input = arguments[0];
            let flatpickrInstance = input._flatpickr;
            
            if (flatpickrInstance) {
                // Clear any existing date
                flatpickrInstance.clear();
                
                // Set the new date with additional options
                flatpickrInstance.setDate(arguments[1], true, 'Y-m-d');
                
                // Trigger multiple events with more specificity
                let events = ['change', 'input', 'blur', 'keyup', 'keydown'];
                events.forEach(eventName => {
                    let event = new Event(eventName, { 
                        bubbles: true, 
                        cancelable: true 
                    });
                    input.dispatchEvent(event);
                });
                
                // Additional check to ensure date is set
                console.log('Set date: ' + flatpickrInstance.selectedDates[0]);
            }
        """, date_input, new_date)
        
        updated_value = date_input.get_attribute("value")
        main = driver.find_element(By.CLASS_NAME, "bk-Column")
        shadow = expand_shadow_element(driver, main)
        
        all_divs = shadow.find_elements(By.CSS_SELECTOR, "div")
        nested_shadow = expand_shadow_element(driver, all_divs[0])
        
        div = nested_shadow.find_elements(By.CSS_SELECTOR, "div")[2]
        shadow = expand_shadow_element(driver, div)

        div = shadow.find_elements(By.CSS_SELECTOR, "div")[1]
        shadow = expand_shadow_element(driver, div)

        div = shadow.find_elements(By.CSS_SELECTOR, "div")[0]
        shadow = expand_shadow_element(driver, div)
        div_2 = div
        
        shadow = expand_shadow_element(driver, div_2)

        div_2 = shadow.find_elements(By.CSS_SELECTOR, "div")[6]
        file_count_before = file_count(download_folder)
        shadow_2 = expand_shadow_element(driver, div_2)
        div_3 = shadow_2.find_elements(By.CSS_SELECTOR, "div")[1]
        shadow_3 = expand_shadow_element(driver, div_3)
        button = shadow_3.find_elements(By.CSS_SELECTOR, "button")[0]
        button.click()
        while file_count(download_folder) == file_count_before:
            time.sleep(1)
        new_name = rename_latest_file(download_folder, f"{new_date}.csv")
        print(f"Downloaded file for {new_date}: {new_name}")

        return True
    
    except Exception as e:
        print(f"Error downloading for {new_date}: {traceback.format_exc()}")
        return False  
    
    
def select_city(driver, city):        
    main = driver.find_element(By.CLASS_NAME, "bk-panel-models-layout-Column")
    shadow = expand_shadow_element(driver, main)
    div = shadow.find_elements(By.CSS_SELECTOR, "div")[1]
    shadow = expand_shadow_element(driver, div)
    input_location = shadow.find_elements(By.CSS_SELECTOR, "input")[0]
    input_location.send_keys(u'\ue003') # Delete
    input_location.send_keys(Keys.CONTROL + "a")  # Select all text
    input_location.send_keys(Keys.DELETE)        # Delete selected text
    input_location.clear()
    input_location.send_keys(city)
    input_location.send_keys(u'\ue007')

def aggregate_city_dataframes(dfs):
    # Ensure all DataFrames have consistent datetime column
    for df in dfs:
        df['datetime (UTC)'] = pd.to_datetime(df['datetime (UTC)'], utc=True)
        df.set_index('datetime (UTC)', inplace=True)
    
    # Identify numeric columns
    numeric_columns = dfs[0].select_dtypes(include=['float64', 'int64']).columns
    
    # Combine and calculate mean for numeric columns
    combined_df = pd.concat(dfs)
    agg_dict = {col: 'mean' for col in numeric_columns}
    
    # Group by datetime and aggregate
    aggregated_df = combined_df.groupby(level=0).agg(agg_dict)
    
    # Reset index to bring datetime back as a column
    aggregated_df.reset_index(inplace=True)
    
    return aggregated_df