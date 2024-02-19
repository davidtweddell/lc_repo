import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import logging


#==============================================================================
def load_dataset_to_df(input_path: str,
                       verbose: bool = False) -> pd.DataFrame:
#==============================================================================
# load a dataset dict and convert it to a dataframe
    
    from project_modules.data_prep import join_datasets

    # load the dataset dict
    if verbose:
        logging.info(f"Loading data from {input_path}...")

    try:
        dsd = DatasetDict.load_from_disk(input_path)
    except FileNotFoundError:
        dsd = Dataset.load_from_disk(input_path)

    # convert to a combined dataframe
    if verbose:
        logging.info("Converting to dataframe...")

    try:
        df = join_datasets(dsd)
    except:
        df = dsd.to_pandas()

    if verbose:
        logging.info(f"Shape: {df.shape}")

    return df
#==============================================================================



