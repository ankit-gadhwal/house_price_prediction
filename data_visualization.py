import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"error occur in data loading {e}")
    
def Scatter_plot(df,output_path):
    plt.figure(figsize = (10,15))

    plt.scatter(
        df['sqft_living'],
        df['price'],
        alpha=0.3,              # 🔥 IMPORTANT
        s=10
    )
    plt.colorbar(label="sqft_above")
    plt.xlabel("sqft_living")
    plt.ylabel("Total Price")
    plt.title("Scatter Plot")

    plt.savefig(output_path)
    plt.close()

def Corelation(df,output_path):
    housing_d = df.select_dtypes(include = [np.number])
    corr_matrix = housing_d.corr()
    
    # 🔥 Only correlation with 'price'
    price_corr = corr_matrix['price'].sort_values(ascending=False)
    # convert to dict
    price_corr = price_corr.to_dict()

    with open(output_path,"w") as f:
        json.dump(price_corr,f,indent = 4)

def main():
    try:
        input_path = "data/processed/train_processed.csv"
        output_dir = "reports"
        plot_path = os.path.join(output_dir,"Plots")
        os.makedirs(plot_path,exist_ok = True)

        df = load_data(input_path)

        # save scatter plot
        Scatter_plot(df,os.path.join(plot_path,"scatter.png"))

        # Save correlation JSON
        Corelation(df,os.path.join(plot_path,"correlation.json"))

    except Exception as e:
        raise Exception(f"Error occured: {e}")

if __name__ =="__main__":
    main()