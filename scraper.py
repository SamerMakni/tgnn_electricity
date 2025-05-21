import argparse, time, json, os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from core import total_load, generator, border
from core.utils import bcolors, VALID_COUNTRIES, VALID_YEARS, URLS

def run_scraper(headless, download_type, login_email, login_password, data_dir, countries, years, action_time):
    print(f'{bcolors.HEADER}Scraper Started!')
    options = Options()
    prefs = {
        "download.default_directory": os.path.abspath(data_dir),
        "download.prompt_for_download": False,
        "profile.default_content_settings.popups": 0,
        "profile.content_settings.exceptions.automatic_downloads.*.setting": 1
    }
    options.add_experimental_option("prefs", prefs)
    if headless:
        options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--remote-debugging-pipe')
    driver = webdriver.Chrome(options=options)

    driver.get(URLS['login'])
    print(f'{bcolors.ENDC} Logging in ...')
    time.sleep(action_time)
    driver.find_element("id", "username").send_keys(login_email)
    time.sleep(action_time)
    driver.find_element("id", "password").send_keys(login_password)
    time.sleep(action_time)
    driver.find_element("name", "login").click()

    driver.implicitly_wait(3)

    if driver.title == 'ENTSO-E Transparency Platform':
        agree_div = driver.find_element('id', 'close-button')
        driver.execute_script("arguments[0].click();", agree_div) 
        print(f'{bcolors.OKGREEN} Successfully logged in')
    else:
        print(f'{bcolors.FAIL} Error at logging in')
    time.sleep(action_time)

    if download_type in ['total_load', 'all']:
        print(f'{bcolors.OKCYAN} Downloading total load started!')
        download_dir = os.path.join(data_dir, 'total_load')
        total_load.download(driver, url=URLS['total_load'], countries=countries, years=years, action_time=action_time, download_dir=download_dir)
    
    if download_type in ['generator', 'all']:
        print(f'{bcolors.OKCYAN} Downloading generator data started!')
        download_dir = os.path.join(data_dir, 'generator')
        generator.download(driver, url=URLS['generator'], countries=countries, years=years, action_time=action_time, download_dir=download_dir)

    if download_type in ['border', 'all']:
        print(f'{bcolors.OKCYAN} Downloading border data started!')
        download_dir = os.path.join(data_dir, 'border')
        border.download(driver, url=URLS['border'], countries=countries, years=years, action_time=action_time, download_dir=download_dir)

    driver.close()
    finish_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    print(f'{bcolors.ENDC} Scraper Finished at {finish_time_str}')

def main(config_path, headless, download, data_dir, countries, years, action_time):
    # load config file
    if not config_path.endswith('.json'):
        raise ValueError('Config file must be a JSON file.')
    with open(config_path) as f:
        config = json.load(f)
    if "email" in config:
        login_email = config['email']
    else:
        raise ValueError('Config file must contain the email field.')
    if "password" in config:
        login_password = config['password']
    else:
        raise ValueError('Config file must contain the password field.')
    
    # run scraper
    run_scraper(
        headless=headless,
        download_type=download,
        login_email=login_email,
        login_password=login_password,
        data_dir=data_dir,
        countries=countries,
        years=years,
        action_time=action_time
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run script headless.')

    parser.add_argument('config_path' ,help='Path to the JSON config file.')
    parser.add_argument('--headless', action='store_true', help='Run the browser in headless mode.')
    parser.add_argument('--download', choices=['total_load', 'generator', 'border', 'all'], default='all',
                        help="Specify which data to download: 'total_load', 'generator', 'border', or 'all'.")
    parser.add_argument('--data_dir', default='./data/raw/', help='Directory to save downloaded files.')
    parser.add_argument('--countries', nargs='+', choices=VALID_COUNTRIES, default=['AL', 'AT', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'GE', 'DE', 'GR', 'HU', 'IE', 'IT', 'XK', 'LV', 'LT', 'LU', 'ME', 'NL', 'MK', 'NO', 'PL', 'PT', 'RO', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'UA', 'UK'], help='Countries for data download.')
    parser.add_argument('--years', nargs='+', choices=VALID_YEARS, default=['2023'], help='Years for data download.')
    parser.add_argument('--action_time', type=int, default=2, help='Time to wait between actions in seconds.')

    args = parser.parse_args()
    main(
        config_path=args.config_path,
        headless=args.headless,
        download=args.download,
        data_dir=args.data_dir,
        countries=args.countries,
        years=args.years,
        action_time=args.action_time
    )