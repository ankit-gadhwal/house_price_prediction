from pathlib import Path
import matplotlib.pyplot as plt 
import pandas as pd 
import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")
    
def load_params(filepath: str) -> float:
    try:
        with open(filepath,"r") as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}:{e}")
    

def split_data(data : pd.DataFrame,test_size: float) ->tuple[pd.DataFrame,pd.DataFrame]: 
    try:
        return train_test_split(data,test_size,random_state= 42)
    except Exception as e:
        raise ValueError(f"Error in splitting data {e}")
    
def save_data(df : pd.DataFrame,filepath: str) -> None:
    try:
        df.to_csv(filepath,index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath} : {e}")
    
def main():
    try:
        data_filepath = "data12shivam.csv"
        params_filepath = "params.yaml"
        raw_data_path = os.path.join("data","raw")
        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data,test_data = split_data(data,test_size)
        os.makedirs(raw_data_path)
        save_data(train_data,os.path.join(raw_data_path,"train_csv"))
        save_data(test_data,os.path.join(raw_data_path,"test_csv"))
    except Exception as e:
        raise Exception(f"An error occured : {e}")