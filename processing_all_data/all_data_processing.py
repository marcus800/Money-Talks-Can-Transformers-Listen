import os
import pandas as pd
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
import pytz
save_location = '../processed_data'


def process_base_data(folder_name,folder_date,save_csv=False):
    with open(f'{folder_name}/syncmap.json', 'r') as file:
        syncmap_data = json.load(file)

    syncmap_list = [{
        'begin': fragment['begin'],
        'end': fragment['end'],
        'sentence': " ".join(fragment['lines']).strip()
    } for fragment in syncmap_data['fragments']]
    syncmap_df = pd.DataFrame(syncmap_list)

    annotations_df = pd.read_csv(f'{folder_name}/Annotations.csv')
    annotations_df['Sentences'] = annotations_df['Sentences'].str.strip()

    # Just for checking prob remove later
    annotations_df['Index'] = annotations_df.index
    for i, row in syncmap_df.iterrows():
        if row['sentence'] != annotations_df.loc[i, 'Sentences']:
            print(f"Mismatch found at index {i}:")
            print(f"  syncmap: {row['sentence']}")
            print(f"  annotations: {annotations_df.loc[i, 'Sentences']}")

    merged_df = pd.merge(syncmap_df, annotations_df, left_index=True, right_index=True)

    monopoly_base_data = merged_df[['begin', 'end', 'Sentences', 'Speaker']]
    monopoly_base_data.columns = ['Start Time', 'End Time', 'Sentence', 'Speaker']
    # print(monopoly_base_data.head())
    conference_start_time = datetime.combine(folder_date, datetime.min.time()) + timedelta(hours=14, minutes=30)

    eastern = pytz.timezone('US/Eastern')
    localized_conference_start_time = eastern.localize(conference_start_time,
                                                       is_dst=None)  # None lets pytz determine if DST is in effect
    utc_conference_start_time = localized_conference_start_time.astimezone(pytz.utc)
    monopoly_base_data = monopoly_base_data.copy()
    monopoly_base_data.loc[:, 'Local Start Time'] = monopoly_base_data['Start Time'].apply(
        lambda x: utc_conference_start_time + timedelta(seconds=float(x)))

    if save_csv:
        monopoly_base_data.to_csv(f"{save_location}/monopoly_base_data.csv", index=False)

    return monopoly_base_data


def soft_match(sentence1, sentence2, threshold=0.2):
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())

    intersection = words1.intersection(words2)
    ratio = len(intersection) / len(words1)

    return ratio >= threshold


def match_and_label_times(folder_name, monopoly_base_data, labeled_file_name, save_csv=False):
    lablled_path = f'{folder_name}/{labeled_file_name}.csv'
    df_labeled = pd.read_csv(lablled_path)

    df_labeled_time = df_labeled.copy()
    # df_labeled_time['Local Start Time'] = np.nan
    # df_labeled_time['Local Start Time'] = df_labeled_time['Local Start Time'].astype(
    #     'datetime64[ns, tz]')  # If converting to timezone-aware
    df_labeled_time['Local Start Time'] = pd.Series([pd.Timestamp(0, tz=pytz.UTC)] * len(df_labeled_time))

    current_final_df_index = 0

    # Also bit slow but whatever
    for index, row in df_labeled_time.iterrows():
        labeled_sentence = row['sentence']

        for time_index in range(current_final_df_index, len(monopoly_base_data)):
            time_sentence = monopoly_base_data.loc[time_index, 'Sentence']

            if soft_match(time_sentence, labeled_sentence):
                start_time = monopoly_base_data.loc[time_index, 'Local Start Time']
                df_labeled_time.at[index, 'Local Start Time'] = start_time
                current_final_df_index = time_index + 1
                break

    # df_labeled_time['Local Start Time'] = pd.to_datetime(df_labeled_time['Local Start Time'], format='%Y-%m-%d %H:%M:%S')
    # df_labeled_time['Local Time'] = pd.to_datetime(
    #     df_labeled_time['Local Time'].dt.tz_localize(pytz.timezone('US/Eastern')), format='%Y-%m-%d %H:%M:%S')

    # Convert label strings to numbers

    # LABEL_2: Neutral
    # LABEL_1: Hawkish
    # LABEL_0: Dovish

    # Lets relaba this

    # hawkish
    # netral 0
    # dov -1

    # When hawkish we expect higher rates => more investment in other currents => usd is weeker, eror is tronger => eurusd down.
    # Other way round when dovish

    relabel_map = {'LABEL_2': 0, 'LABEL_1': 1, 'LABEL_0': -1}
    df_labeled_time['label'] = df_labeled_time['label'].apply(lambda x: relabel_map[x])
    df_labeled_time.rename(columns={"sentence":"Sentence"},inplace=True)

    if save_csv:
        df_labeled_time.to_csv(f"{save_location}/df_labeled_time.csv", index=False)

    return df_labeled_time


def parse_datetime_with_timezone(dt_str):
    # Extract the base datetime and timezone parts
    base_dt, tz_offset = dt_str.rsplit(' ', 1)
    # Parse the datetime without timezone
    dt = datetime.strptime(base_dt, '%d.%m.%Y %H:%M:%S.%f')
    # Parse the timezone offset
    sign = tz_offset[3]  # + or -
    hours_offset = int(tz_offset[4:6])
    minutes_offset = int(tz_offset[6:8])
    # Calculate total offset in minutes
    total_offset = hours_offset * 60 + minutes_offset
    if sign == '-':
        total_offset = -total_offset
    # Create timezone object
    tz = pytz.FixedOffset(total_offset)
    # Localize the datetime object with the parsed timezone
    dt = dt.replace(tzinfo=tz)
    return dt



def process_price_data(folder_name, file_price_data, data_type, formatted_date, save_csv=False):
    # TODO! COME back, for now using bid prices from tick data resampled.
    file_path = f'{folder_name}/{file_price_data}/{data_type}_Ticks_{formatted_date}-{formatted_date}.csv'
    df = pd.read_csv(file_path)
    df.rename(columns={'Local time': 'Local Time'}, inplace=True)

    # Apply the custom parsing function to each datetime string
    df['Local Time'] = df['Local Time'].apply(parse_datetime_with_timezone)

    df['Local Time'] = df['Local Time'].dt.tz_convert('UTC')

    df.set_index('Local Time', inplace=True)
    df = df['Bid'].resample('1s').ohlc()
    df_ffilled = df.ffill()


    df_ffilled.columns = ['Open', 'High', 'Low', 'Close']

    #Not gonna bother filtering anymore, no need
    # start_time = pd.to_datetime('2022-01-26 13:00').time()
    # end_time = pd.to_datetime('2022-01-26 17:00').time()
    # df_filtered = df_ffilled[(df_ffilled.index.time >= start_time) & (df_ffilled.index.time <= end_time)]



    if save_csv:
        df_ffilled.to_csv(f"{save_location}/{data_type}_filtered.csv", index=True)

    return df_ffilled



def example_case_types(df_input, map_of_prices, name, delay=2, save_csv=False):
    eval_dataset = df_input.copy()

    # Round the local times to the nearest second
    eval_dataset['Rounded Start Local Time'] = pd.to_datetime(eval_dataset['Local Start Time']).dt.round('s')
    eval_dataset['Rounded End Local Time'] = pd.to_datetime(eval_dataset['Local Start Time']).dt.round('s').shift(-1)
    for k,v in map_of_prices.items():
        df_price_type_filtered_eval = v.copy()
        df_price_type_filtered_eval['Rounded Local Time'] = df_price_type_filtered_eval.index.round('s')
        df_price_type_filtered_eval['Delayed Rounded Local Time'] = df_price_type_filtered_eval[
                                                                    'Rounded Local Time'] + pd.Timedelta(seconds=delay)

        close_prices_map_delayed = df_price_type_filtered_eval.set_index('Delayed Rounded Local Time')['Close'].to_dict()
        close_prices_map_non_delayed = df_price_type_filtered_eval.set_index('Rounded Local Time')['Close'].to_dict()


        eval_dataset[f'Closest {k} End Close Delayed'] = eval_dataset['Rounded End Local Time'].map(close_prices_map_delayed)
        eval_dataset[f'Closest {k} Start Close non Delayed'] = eval_dataset['Rounded Start Local Time'].map(close_prices_map_non_delayed)
        eval_dataset[f'{k} Close Difference'] = eval_dataset[f'Closest {k} End Close Delayed'] - eval_dataset[f'Closest {k} Start Close non Delayed']

    if save_csv:
        eval_dataset.to_csv(f"{save_location}/example_{name}_with_delay_{delay}.csv", index=False)

    return eval_dataset


def process_all_folders(df_folders):
    day_to_dfs = {}
    for index, row in df_folders.iterrows():
        folder_name = row['Folder Containing new_price_data']

        # Assuming each folder follows a specific naming convention for date
        # e.g., "January_26_2022" or "2022-01-26"
        # You might need to adjust the logic below based on your actual folder name format
        # This example assumes folder names are in "YYYY-MM-DD" format
        date_part = folder_name.split("\\")[2].split('_')[0]  # This should give "April 25, 2012"
        try:
            print(date_part)
            folder_date = datetime.strptime(date_part, "%B %d, %Y")
        except ValueError as e:
            print(f"Skipping {folder_name} due to date parsing error: {e}")
            continue
        formatted_date = folder_date.strftime("%d.%m.%Y")

        file_price_data = 'new_price_data'
        data_types = ["EURUSD", "USA500.IDXUSD"]
        labeled_file_name = "labeled_FOMCpresconf" + folder_date.strftime("%Y%m%d") + "_select_filtered"

        monopoly_base_data = process_base_data(folder_name,folder_date, save_csv=False)
        df_labeled_time = match_and_label_times(folder_name, monopoly_base_data, labeled_file_name, save_csv=False)

        price_data_map = {}
        for data_type in data_types:
            price_data_map[data_type] = process_price_data(folder_name, file_price_data, data_type, formatted_date,
                                                       save_csv=False)
        labled_processed = example_case_types(df_labeled_time, price_data_map, name=f"labled", delay=2, save_csv=False)
        base_processed = example_case_types(monopoly_base_data, price_data_map, name=f"base", delay=2, save_csv=False)

        day_to_dfs[date_part] = [labled_processed,base_processed]
    return day_to_dfs

data_path = "..\data"
folders_with_new_price_data = []

for root, dirs, files in os.walk(data_path):
    if 'new_price_data' in dirs:
        folders_with_new_price_data.append(root)

df_folders = pd.DataFrame(folders_with_new_price_data, columns=['Folder Containing new_price_data'])
print(df_folders)

# df_folders = df_folders[:1]
# Now process all folders
map = process_all_folders(df_folders)
# print(map)
def save_processed_data(day_to_dfs, base_path):
    os.makedirs(base_path, exist_ok=True)

    labeled_dir = os.path.join(base_path, 'labeled')
    monopoly_dir = os.path.join(base_path, 'monopoly')

    os.makedirs(labeled_dir, exist_ok=True)
    os.makedirs(monopoly_dir, exist_ok=True)

    for day, dfs in day_to_dfs.items():
        # date_str = day.split('/')[-1]
        date_str = day
        # print(date_str)

        labeled_file_path = os.path.join(labeled_dir, f'{date_str}.csv')
        monopoly_file_path = os.path.join(monopoly_dir, f'{date_str}.csv')

        if len(dfs) > 0:
            dfs[0].to_csv(labeled_file_path, index=True)
        if len(dfs) > 1:
            dfs[1].to_csv(monopoly_file_path, index=True)


save_processed_data(map,save_location)
