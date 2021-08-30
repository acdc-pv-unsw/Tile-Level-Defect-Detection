# import libraries
from sklearn.utils import resample
import pandas as pd

# Define downsampling of the dataframe
def Downsizing(dataframe,random_seed=None):
    # Separate majority and minority classes
    df_majority = dataframe[dataframe.Label == "No Anomaly"]
    df_minority = dataframe[dataframe.Label != "No Anomaly"]

    # Downsample majority class to size of the minority class
    downsize = df_minority.shape[0]

    # sample without replacement replace = false
    # to match minority class n_samples = downsize[0]
    # reproducible results with random_state = int
    df_majority_downsampled = resample(df_majority, replace=False,n_samples=int(downsize*0.7),random_state=random_seed)
    # Combine minority class with downsampled majority class
    return pd.concat([df_majority_downsampled, df_minority])
