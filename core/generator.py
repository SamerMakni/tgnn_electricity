from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from core.utils import bcolors, select_file_name, Spinner

def download(driver, url, countries, years, action_time, download_dir):
    countries = countries
    driver.get(url)
    driver.implicitly_wait(1)
    
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "dv-market-areas-content")))
    time.sleep(action_time)
    for country in countries:
        spinner = Spinner(f"{bcolors.ENDC}  Downloading data for {country} ... ")
        spinner.start()
        for year in years:
            try:
                driver.get(f'{url}?dateTime.dateTime=01.01.{year}+00:00|CET|DAYTIMERANGE&dateTime.endDateTime=01.01.{year}+00:00|CET|DAYTIMERANGE')
                time.sleep(action_time)
                country_tab = driver.find_element(By.XPATH, "//li[contains(text(), 'Country')]")
                driver.execute_script("arguments[0].click();", country_tab)
                label_text = f"({country})"
                label = driver.find_element(By.XPATH, f"//label[contains(text(), '{label_text}')]")
                driver.execute_script("arguments[0].click();", label)
                time.sleep(action_time)
                export_button = driver.find_element(By.ID, "dv-export-data")
                driver.execute_script("arguments[0].scrollIntoView(true);", export_button)
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "dv-export-data")))
                driver.execute_script("arguments[0].click();", export_button)
                time.sleep(action_time)
                
                download_button = driver.find_element(By.XPATH, f"//a[contains(text(), 'Actual Generation per Production Type (Year, CSV)')]") 
                driver.execute_script("arguments[0].click();", download_button)
                time.sleep(action_time + 3)
                print()
                select_file_name(f"generator_{country}_{year}.csv", download_dir)

            except Exception as e:
                spinner.stop()
                print(f"{bcolors.FAIL}Error for country: {country}, year: {year}. Error: {str(e)}")
        print()
        spinner.stop()
        print(f"{bcolors.OKGREEN} Download complete for {country}")

