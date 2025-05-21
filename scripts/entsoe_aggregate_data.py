import pandas as pd
import os
import glob
import argparse
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
from parameters import country_borders


def clean_date_column(df, column_name, type=0):
    if type == 0:
        df[column_name] = df[column_name].str.split(' - ').str[0]
        df[column_name] = pd.to_datetime(df[column_name], format='%d.%m.%Y %H:%M')
    elif type == 1:
        # 2023-01-01 00:00:00+00:00, remove utc
        df[column_name] = df[column_name].str.split('+').str[0] 
        df[column_name] = pd.to_datetime(df[column_name], format='%Y-%m-%d %H:%M:%S')        
    return df

def resample_to_hourly(df, area_column):
    df['MTU'] = pd.to_datetime(df['MTU'])
    df.set_index('MTU', inplace=True)

    cols = df.columns.difference([area_column])
    cols_to_sum = cols[:-2]
    cols_to_mean = cols[-2:]

    df_resampled_sum = df[cols_to_sum].resample('H').sum()
    df_resampled_mean = df[cols_to_mean].resample('H').mean()

    df_resampled = pd.concat([df_resampled_sum, df_resampled_mean], axis=1)
    df_resampled = df_resampled.reindex(columns=cols, fill_value=0)
    
    df_resampled[area_column] = df[area_column].resample('H').first()

    df_resampled.columns = df_resampled.columns.str.replace(r'Day-ahead Total Load Forecast \[MW\] - .*', 'Day-ahead Total Load Forecast [MW]', regex=True)
    df_resampled.columns = df_resampled.columns.str.replace(r'Actual Total Load \[MW\] - .*', 'Actual Total Load [MW]', regex=True)
    return df_resampled


def combine_border_files(country_borders, border_location, year, output_dir):
    all_data = {}

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the country_borders dictionary to find all possible CSV files
    for country_code, (neighbors, country_name, _, country_shortcode) in country_borders.items():
        for neighbor_code in neighbors:
            neighbor_shortcode = country_borders[neighbor_code][3]
            c = country_name.split("(")[0].strip().replace(' ', '_')
            
            # Construct the CSV filename
            csv_filename = f"border_{c}_{country_borders[neighbor_code][1].replace(' ', '_')}_{year}.csv"
            csv_path = os.path.join(border_location, csv_filename)
            
            # Check if the CSV file exists
            if os.path.exists(csv_path):
                        try:
                            # Load the CSV file
                            df = pd.read_csv(csv_path)
                            
                            # Convert the Time (UTC) column to datetime
                            
                            df['Time (UTC)'] = df['Time (UTC)'].str[:16]
                            df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], format='%d.%m.%Y %H:%M')
                            
                            # Convert relevant columns to numeric, coercing errors to NaN
                            df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # Column 1: Country A > Country B Total [MW]
                            df.iloc[:, 3] = pd.to_numeric(df.iloc[:, 3], errors='coerce')  # Column 3: Country B > Country A Total [MW]
                            
                            # Calculate the net flow (edge weight)
                            df['Net Flow'] = df.iloc[:, 1] - df.iloc[:, 3]
                            
                            # Aggregate hourly
                            df = df.resample('H', on='Time (UTC)').sum().reset_index()
                            print(df.shape)
                            # Store the aggregated data in the dictionary
                            all_data[f"{country_shortcode}_{neighbor_shortcode}"] = df[['Time (UTC)', 'Net Flow']]
                            
                            print(f"Processed: {csv_filename}")
                        except Exception as e:
                            print(f"Error processing {csv_filename}: {e}")
            else:
                print(f"File not found: {csv_filename}")

    # Combine all data into a single DataFrame
    combined_df = None
    for key, df in all_data.items():
        df = df.rename(columns={'Net Flow': key})  # Rename the Net Flow column to the country pair
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='Time (UTC)', how='outer')

    # Save the combined DataFrame to a single CSV file
    output_path = os.path.join(output_dir, "all_net_flows_2023.csv")
    combined_df.to_csv(output_path, index=False)
    

def combine_price_files(country_borders, price_location, year, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    
    pass


def zeros_df():
    # generate a dataframe with two columns, one for the time and one for the price
    # the price column will be filled with zeros
    df = pd.DataFrame(columns=['Datetime (UTC)', 'Price (EUR/MWhe)'])
    df['Datetime (UTC)'] = pd.date_range(start='2015-01-01', end='2023-12-31 23:00:00', freq='H')
    
    return df

def main(generator_location, total_load_location, output_location, weather_location, border_location, price_location, year):
    generator_files = glob.glob(f'{generator_location}generator_*_{year}.csv')
    total_load_files = glob.glob(f'{total_load_location}total_load_*_{year}.csv')
    weather_files = glob.glob(f'{weather_location}weather_*_{year}.csv')
    price_files = glob.glob(f'{price_location}*.csv')

    data_frames = []

    for gen_file in generator_files:
        country_code = gen_file.split('_')[2]

        load_file = f'{total_load_location}total_load_{country_code}_{year}.csv'
        weather_file = f'{weather_location}weather_{country_code}_{year}.csv'
        price_file = f'{price_location}{country_code}.csv'
        
        if weather_file not in weather_files:
            print(f"No matching weather file found for {country_code}")
            continue
        
        if load_file not in total_load_files:
            print(f"No matching load file found for {country_code}")
            continue

        if price_file not in price_files:
            print(f"No matching price file found for {country_code}, all prices will be set to 0")
            price_file = None

        gen_df = pd.read_csv(gen_file)
        load_df = pd.read_csv(load_file)
        weather_df = pd.read_csv(weather_file)
        price_df = pd.read_csv(price_file) if price_file else zeros_df()
        

        gen_df = clean_date_column(gen_df, 'MTU')
        load_df = clean_date_column(load_df, 'Time (UTC)')
        weather_df = clean_date_column(weather_df, 'datetime (UTC)', 1)     
        weather_df = weather_df.drop(columns=['utc_offset (hrs)', 'model elevation (surface)'])
        price_df = clean_date_column(price_df, 'Datetime (UTC)', 1) if price_file else price_df
        price_df = price_df.drop(columns=['Country', 'ISO3 Code', 'Datetime (Local)']) if price_file else price_df
              
        
        gen_df.fillna(0)
        load_df.fillna(0)
        weather_df.fillna(0)
        price_df.fillna(0)

        gen_df.replace(r'^\s*$', 0, regex=True)
        load_df.replace(r'^\s*$', 0, regex=True)
        weather_df.replace(r'^\s*$', 0, regex=True)
        price_df.replace(r'^\s*$', 0, regex=True)
        
        merged_df = pd.merge(gen_df, load_df, left_on='MTU', right_on='Time (UTC)', how='outer')
        merged_df = pd.merge(merged_df, weather_df, left_on='MTU', right_on='datetime (UTC)', how='outer')
        merged_df = pd.merge(merged_df, price_df, left_on='MTU', right_on='Datetime (UTC)', how='outer')
        

        merged_df['day sin'] = np.sin(merged_df['MTU'].dt.dayofyear * (2 * np.pi / 365))
        merged_df['day cos'] = np.cos(merged_df['MTU'].dt.dayofyear * (2 * np.pi / 365))
        merged_df['hour sin'] = np.sin(merged_df['MTU'].dt.hour * (2 * np.pi / 24))
        merged_df['hour cos'] = np.cos(merged_df['MTU'].dt.hour * (2 * np.pi / 24))
        merged_df['weekday sin'] = np.sin(merged_df['MTU'].dt.weekday * (2 * np.pi / 7))
        merged_df['weekday cos'] = np.cos(merged_df['MTU'].dt.weekday * (2 * np.pi / 7))
        merged_df.drop(columns=['Time (UTC)'], inplace=True)
        merged_df.drop(columns=['datetime (UTC)'], inplace=True)

        
        df_resampled = resample_to_hourly(merged_df, 'Area')
        df_resampled = df_resampled.reset_index()

        
        data_frames.append(df_resampled)
        

        output_file = f'{output_location}/{country_code}_{year}_data.csv'
        df_resampled.to_csv(output_file, index=False)
        print(f"Data for {country_code} saved to {output_file}")
    combine_border_files(country_borders, border_location, year, output_location)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process generator and total load data.')
    
    parser.add_argument('--generator_location', type=str, default='/.generator/', help='Path to generator data files')
    parser.add_argument('--total_load_location', type=str, default='/.total_load/', help='Path to total load data files')
    parser.add_argument('--border_location', type=str, default='/.border/', help='Path to border data files')
    parser.add_argument('--output_location', type=str, default='.aggregated', help='Directory where output files will be saved')
    parser.add_argument('--weather_location', type=str, default='.weather/2023_bu/', help='Directory where weather files are saved')
    parser.add_argument('--price_location', type=str, default='.prices/', help='Directory where price files are saved')
    parser.add_argument('--year', type=str, default='2023', help='Data year')

    args = parser.parse_args()

    main(args.generator_location, args.total_load_location, args.output_location, args.weather_location, args.border_location, args.price_location, args.year)