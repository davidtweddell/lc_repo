import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

#==============================================================================
def join_datasets(dsd: DatasetDict) -> pd.DataFrame:
#==============================================================================


    # make a dataframe to hold the joined datasets
    df = pd.DataFrame()

    # loop through the datasets in the dataset dict
    for ds in dsd.values():

        # concatente the dataset into the dataframe
        df = pd.concat([df, ds.to_pandas()], axis=1)

    return df

