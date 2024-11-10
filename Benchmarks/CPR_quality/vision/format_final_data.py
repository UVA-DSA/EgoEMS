import os
import pandas as pd

FRAME_RATE = 30

FINAL_COLUMNS = [
    'Subject',
    'GT Cycles',
    'Est Cycles',
    'Cycle Error',
    'GT CPR cycles per min',
    'Predicted CPR cycles per min',
    'Cycle Error per min',
    'GT Depth',
    'Est Depth',
    'Depth Error'
]

def calculate_error(final_data):
    # Calculate average cycle error
    final_data['Cycle Error'] = (final_data['GT Cycles'] - final_data['Est Cycles']).abs()
    
    # Calculate average GT and Predicted CPR cycles per minute
    final_data['GT CPR cycles per min'] = (final_data['GT Cycles'] / (((final_data['End Frame'] - final_data['Start Frame']) / FRAME_RATE)) / 60)
    final_data['Predicted CPR cycles per min'] = (final_data['Est Cycles'] / (((final_data['End Frame'] - final_data['Start Frame']) / FRAME_RATE)) / 60)
    final_data['Cycle Error per min'] = (final_data['GT CPR cycles per min'] - final_data['Predicted CPR cycles per min']).abs()

    # Calculate average depth error
    final_data['Depth Error'] = (final_data['GT Depth'] - final_data['Est Depth']).abs()
    
    return final_data

if __name__ == "__main__":
    dfs = []

    # Read all CSV files in the directory
    for file in os.listdir('/home/cogems_nist/repos/EgoExoEMS'):
        if file.endswith('.csv'):
            df = pd.read_csv(file)

            # Strip leading/trailing whitespace from column headers
            df.columns = df.columns.str.strip()

            dfs.append(df)

    # Concatenate all dataframes into one
    final_data = pd.concat(dfs, ignore_index=True)

    # Coerce 'Est Depth' to float, invalid parsing will be set as NaN
    final_data['Est Depth'] = pd.to_numeric(final_data['Est Depth'], errors='coerce')

    # Drop rows with NaN values after coercion
    final_data.dropna(inplace=True)

    # Enrich the dataframe with calculated averages and errors
    enriched_final_data = calculate_error(final_data)
    enriched_final_data = enriched_final_data[FINAL_COLUMNS]

    # Average across unique subjects
    subject_averages = enriched_final_data.groupby('Subject').mean().reset_index()
    print(subject_averages)

    subject_averages.to_csv('/home/cogems_nist/repos/EgoExoEMS/final_data_average.csv', index=False)

