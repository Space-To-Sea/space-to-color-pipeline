from enum import Enum
import pandas as pd
import glob
import re
import os
import numpy as np
from pathlib import Path

folder_path = r"F:/UROP/download_test/"
output_path= r"C:/Users/raine/Data/School/MIT/Freshman Year/UROP/CSV/lyze/samples/2-17/"
output_csv_filename=r"debugging.csv"
class DataType(Enum):
    #the variable names in the CSV file
    CHLOR = "chlor_a"
    DIAT = "diatoms_hirata"
    DINO = "dinoflagellates_hirata"
    GREEN = "greenalgae_hirata"
    PRYM = "prymnesiophytes_hirata"
    
def get_dates(csv_paths: list[str]) -> list[str]:
    """Retrieve dates from file path names
    
    Args:
        csv_paths (List[str]): list of CSV file paths
    
    Returns:
        List[str]: List of dates extracted from the file paths (format: 'YYYY-MM-DD').
    """
    dates = []
    for csv_path in csv_paths:
        date=re.search(r'(\d{4}-\d{2}-\d{2})',csv_path) #expects file path containing {YYYY}-{MM}-{DD}.csv
        if not date:
            raise ValueError(csv_path+"does not contain expected pattern: YYYY-MM-DD.csv")
        dates.append(date.group(0)) 
    return dates

def read_csv(csv_path: str) -> pd.DataFrame:
    """Reads csv file as dataframe object
    """
    
    #search for the row that contains all the variable names
    with open(csv_path, 'r') as f:
        numLinesBeforeHeader=0  
        for line in f:
            if line.startswith('featureId'): 
                break
            else:
                numLinesBeforeHeader+=1
        else:
            raise ValueError(f"Header row starting with 'featureId' not found in {csv_path}")
    
    #read in the csv file, starting with the row that contains all the variable names
    df = pd.read_csv(csv_path,sep=' ',skiprows=numLinesBeforeHeader,header=0,comment='#',on_bad_lines='skip')
    df.columns=[c.split(':')[0] for c in df.columns]
    print(f"Read {csv_path}")
        
    return df

def convert_percentages_to_concentrations(df: pd.DataFrame):
    """
    Convert phytoplankton percentage columns in the DataFrame
    to absolute chlorophyll concentrations using total chlorophyll,
    and replace negative values with NaN.

    Args:
        df (pd.DataFrame): DataFrame containing columns for total chlorophyll
                           and plankton percentages (DataType enum).

    Returns:
        pd.DataFrame: The same DataFrame with plankton concentrations updated.
    """
    for data_type in DataType:
        if data_type!=DataType.CHLOR:
            df[data_type.value]=df[data_type.value]*df[DataType.CHLOR.value]
        df.loc[df[data_type.value] < 0, data_type.value] = np.nan
    return df

def calc_minmax(region_df: pd.DataFrame):
    """
    Compute the minimum and maximum values for each DataType column
    in a regional DataFrame.

    Parameters
    ----------
    region_df : pandas.DataFrame
        DataFrame containing numeric columns corresponding to each
        member of the DataType enum. Column names must match
        `DataType.value`.

    Returns
    -------
    dict[DataType, list[float, float]]
        Dictionary mapping each DataType enum member to a
        [min_value, max_value] list. NaN values are ignored
        when computing extrema.
    """
    minmaxdict={}
    for data_type in DataType:
        minmaxdict[data_type] = [
            region_df[data_type.value].min(skipna=True),
            region_df[data_type.value].max(skipna=True)
        ]       
        
        assert minmaxdict[data_type][1]>=minmaxdict[data_type][0], "Error. Max is smaller than min"
    return minmaxdict

def extract_data(df: pd.DataFrame, date):
    """
   Aggregate phytoplankton data by region for a given date.

   For each region (1–4), this function computes summary statistics 
   (average, maximum, and minimum) for each datatype specified 
   in the `DataType` enum. The results are returned as a list of dictionaries, 
   one dictionary per region.

   Args:
       df (pd.DataFrame): DataFrame containing phytoplankton data. 
           Must have a 'region' column and columns corresponding to 
           all members of the `DataType` enum.
       date (str or datetime): The date associated with the data.

   Returns:
       list of dict: A list containing one dictionary per region, where each 
       dictionary includes:
           - 'date': the date provided
           - 'region': the region number (as float)
           - '<data_type>_avg': average concentration of that type
           - '<data_type>_max': maximum subregion avg concentration of that type
           - '<data_type>_min': minimum subregion avg concentration of that type
         for all datatypes in `DataType`.
    """
    data=[]
    for region_num in range(1,5,1):
        mask=(df['region']==region_num)
        region_df=df.loc[mask]
        #selects only the data corresponding to the region and datatypes in DataType
        
        if region_df.empty:
            continue

        fraction_null = region_df['chlor_a'].isna().mean()
        if fraction_null>=.9:
            #if greater than 90% of the region are null values
            #(usually due to heavy cloud cover)
            #save this region's data as nan
            region_data={}
            for data_type in DataType:
                region_data[data_type.value+"_avg"]=np.nan
                region_data[data_type.value+"_max"]=np.nan
                region_data[data_type.value+"_min"]=np.nan
    
            row = {
                'date': date,
                **region_data,
                'region': float(region_num)
            }
            data.append(row)
            continue
        
        minmaxdict=calc_minmax(region_df)
        
        region_data={}
        for data_type in DataType:
            region_data[data_type.value+"_avg"]=region_df[data_type.value].mean()
            region_data[data_type.value+"_max"]=minmaxdict[data_type][1]
            region_data[data_type.value+"_min"]=minmaxdict[data_type][0]

        row = {
            'date': date,
            **region_data,
            'region': float(region_num)
        }
        data.append(row)
    return data

def write_csv_file(data, path):
    """Write CSV data to file"""
    if not data:
        return
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # creates dirs if needed

    df = pd.DataFrame(data)
    
    variables=[]
    for data_type in DataType:
        variables.append(data_type.value+"_avg")
        variables.append(data_type.value+"_max")
        variables.append(data_type.value+"_min")

    column_order = ['date'] + variables + ['region']
    df = df[column_order]

    header_row = "date " + " ".join([f"{var}:float" for var in variables])+ " region:float"
    # print(df)
    with open(path, 'w') as f:
        f.write(f"{header_row}\n")
        df.to_csv(f, sep=' ', index=False, header=False, float_format='%.8f')

if __name__ == '__main__':
    csv_paths = glob.glob(folder_path + "*.csv")  # Find all .csv files in the folder
    print("Found CSV files:", csv_paths)
    
    if csv_paths:  # Check if any CSVs were found
        dates = get_dates(csv_paths)
        
        #sort the csvs by date
        dates_sorted, csv_paths_sorted = zip(*sorted(zip(dates, csv_paths)))
        dates_sorted = list(dates_sorted)
        csv_paths_sorted = list(csv_paths_sorted)
        
        data=[]
        for i,csv_path in enumerate(csv_paths_sorted):
            csv_path = Path(csv_path)  # normalize path
            df=read_csv(csv_path) #read in the csv into a pandas dataframe object
            df=convert_percentages_to_concentrations(df) #convert percentages of chlorophyll to concentration of chlorophyll
            data.extend(extract_data(df,dates_sorted[i])) #reformat the df as a dict with avg, max, min values for each datatype
        
        csv_save_path = os.path.join(output_path, output_csv_filename)
        write_csv_file(data,csv_save_path)
        
    else:
        print("No CSV files found in the folder.")