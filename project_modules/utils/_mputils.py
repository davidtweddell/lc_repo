import os
from copy import deepcopy
import pandas as pd
class MPutils:
    # define a class to time code exeuction using
    # context manager `with` statement
    def get_saving_dir(dir):
        try:
            if os.path.isdir(dir):
                return dir
            else:
                os.makedirs(dir)
            return dir
        except FileExistsError:
            return dir
        
    def reorder_columns(df:pd.DataFrame,firstCols:list):
        df = deepcopy(df)
        lCols = deepcopy(list(df.columns))
        for col in firstCols:
            lCols.remove(col)
        lCols = firstCols+ lCols
        return df[lCols]