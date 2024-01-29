import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# load sklearn label encoder and onehot encoder
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler,
                                   MultiLabelBinarizer, OrdinalEncoder)

DEBUG    = False
INT_TYPE = np.int64

#----------------------------------------
def label_encoder(
                    df: pd.DataFrame, 
                    verbose: bool = False
                    ):
#----------------------------------------

    # instantiate an ordinal encoder
    oe = OrdinalEncoder(dtype = INT_TYPE)
    oe.set_output(transform = "pandas")

    # - the dataframe should be passed with UUID as index
    # - the dataframe should not include any cols that need
    # special multilabel encoding

    # split into object cols for encoding, and numeric cols to be left alone
    o_df = df.select_dtypes(include="object")
    n_df = df.select_dtypes(exclude="object")

    if verbose | DEBUG:
        print(f"Object  o_df: {o_df.shape}")
        print(f"Numeric n_df: {n_df.shape}")

        print(f"o_df: {o_df.columns}")
        print(f"n_df: {n_df.columns}")

    # encode the object columns
    encoded_df = oe.fit_transform(o_df)
    
    # join the encoded object columns with the numeric columns
    df = encoded_df.join(n_df)

    if verbose | DEBUG:
        print(f"Result df: {df.shape}")
        print(f"Result df: {df.columns}")

        print(f"oe.categories_:       {oe.categories_}")
        print(f"oe.feature_names_in_: {oe.feature_names_in_}")

    return df, oe
    

#----------------------------------------
def multilabel_encoder(
                        df: pd.DataFrame, 
                       c: str, 
                       tag: str, 
                       verbose: bool = False,
                        classes  = None
                        ):
#----------------------------------------
    
    print(f'>>> ... encoding multilabel column {c}')

    mlb = MultiLabelBinarizer(classes = classes, sparse_output = False)

    # cast to string to be sure
    df[c] = df[c].astype(str)
    
    # split the items into lists and strip whitespace from each item
    df[c] = df[c].apply(lambda x: [y.strip() for y in x.split(",")])

    # fit
    mlb.fit(df[c])

    # transform
    ff = mlb.transform(df[c])

    # make column names
    if tag == None:
        col_names = [f"{c}_{x}" for x in mlb.classes_]
    else:
        col_names = [f"{tag}_{c}_{x}" for x in mlb.classes_]

    # make a new df with the transformed data
    new_df = pd.DataFrame(ff, columns=col_names)
    new_df.index = df.index

    # TODO: add drop none option

    try:
        # drop the column with "none"
        new_df.drop(f"{c}_none", axis=1, inplace=True)
    except KeyError:
        pass

    if verbose | DEBUG:
        print(new_df.columns)

    return new_df, mlb


#----------------------------------------
def invert_encode(
                df: pd.DataFrame, 
                px: int, 
                encoder_dict: dict,
                ):
#----------------------------------------

    for col in df.columns:
        if col in encoder_dict.keys():

            enc = encoder_dict[col]

            # check the type of the encoder
            if isinstance(enc, OrdinalEncoder):
                print(f"{enc}: {col:<20s} {enc.inverse_transform([[df.loc[px][col]]])[0]}")

            elif isinstance(enc, MultiLabelBinarizer):
                pass

            else:
                print(f"ERROR: encoder type not recognized: {type(enc)}")


#----------------------------------------
def plot_df(
                df: pd.DataFrame, 
                annotate: bool = False,
                scale: bool = True, 
                figsize: tuple = (8,8), 
                verbose: bool = False, 
                ):
#----------------------------------------
    # drop object type columns
    object_cols = [col for col in df.columns if df[col].dtype in ["object", "datetime64[ns]"]]

    df_plot=  df.drop(columns=object_cols)

    if scale == True:
        # scale for viz
        # we can recover the min, max, scale, etc to reconstruct the data
        scaler = MinMaxScaler()
        scaler.fit(df_plot)
        data = scaler.transform(df_plot)
    else:
        data = df_plot

    df_plot = pd.DataFrame(data, columns=df_plot.columns, index=df_plot.index)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
                df_plot.T, 
                cbar       = False,
                ax         = ax, 
                linewidths = 0.5, 
                annot      = annotate, 
                fmt        = "0.1f", 
                cmap       = "viridis"
                )


#---------------------------------
def map_to_tuple(d):
#---------------------------------
# map a column of strings to a tuple
#---------------------------------
    
    if d == "":
        return tuple()

    else:
        try:
            t1 = [int(x) for x in str(d).split(",")]
        except ValueError:
            t1 = [int(d)]
        t2 = tuple(t1)
        return t2


#----------------------------------------
def encode_features_1(
                        df: pd.DataFrame, 
                        mc: pd.DataFrame, 
                        tag: str, 
                        verbose: bool = True
                      ):
#----------------------------------------

    # remove the columns we want to treat as multilabel

    for_multilabel = df[mc]

    # drop the multilabel columns
    print(f'>>> ... deferring {len(mc)} multilabel column(s): {mc}')
    df.drop(mc, axis=1, inplace=True)

    # do the ordinal encoding
    print(f'>>> ... encoding {len(df.columns)} columns')
    df, encoder = label_encoder(df, verbose=verbose)

    # rename columns by prepending a tag
    df.columns = [f'{tag}_{col.lower()}' for col in df.columns]
    #====================

    return df, encoder


#----------------------------------------
def encode_features(
                        df: pd.DataFrame, 
                        tag: str, 
                        verbose: bool = True
                     ):
#----------------------------------------

    # do the ordinal encoding
    print(f'>>> ... encoding {len(df.columns)} columns')
    df, encoder = label_encoder(df, verbose=verbose)

    # rename columns by prepending a tag
    df.columns = [f'{tag}_{col.lower()}' for col in df.columns]
    #====================

    return df, encoder



#----------------------------------------
def read_data(
                fn: str,
                sheet: str, 
                index_col: str = None, 
                verbose: bool = False
                ):
#----------------------------------------

    print(f'>>> Reading data from {fn},  sheet {sheet}')

    # get the file type by splitting the filename
    file_type = fn.split(".")[-1]


    if file_type == "csv":
        df = pd.read_csv(fn, index_col=index_col, sheet_name = sheet).fillna("")

    elif file_type == "xlsx":
        df = pd.read_excel(fn, index_col=index_col, sheet_name = sheet).fillna("")

    else:
        print(f"ERROR: file type not recognized: {file_type}")
        df =  None

    if index_col is not None:
        print(f'>>> Initial shape:   {df.shape}')
        print(f'>>> Index set using: {index_col}')
    else:
        print(f'>>> Initial shape: {df.shape}')

    return df

#----------------------------------------
def summarize_df(
                df: pd.DataFrame, 
                verbose: bool = False,
                ):
#----------------------------------------
    # count the number of different dtypes in the dataframe
    print(f'-'*50)
    print(f">>> DTYPE SUMMARY: {len(df.columns)} columns")
    print(f'-'*50)

    # # count dtypes in the df
    # dtypes = df.dtypes.value_counts()
    # print(f'>>> DTYPE SUMMARY: \n{dtypes}\n')

    from collections import Counter
    cc = Counter(dict(df.dtypes).values())

    #format the counter nicely and print it
    print(f'>>> Column type counts')
    print(f'-'*50)
    for k,v in cc.items():
        print(f"{str(k):>20s}: {v:>4d}")

    # columns
    if verbose | DEBUG:
        for item in df.columns:
            print(f'>>> {item:30s} dtype: {str(df[item].dtype):<20s}unique values in: {df[item].nunique():>5d}')


#----------------------------------------
def summarize_encoder(
                        e_dict: dict, 
                        verbose: bool = False
                        ):
#----------------------------------------

    for k in e_dict.keys():
        e = e_dict[k]

        if type(e) == OrdinalEncoder:
            cc = e.categories_
            print(f'>>> OrdinalEncoder {k}: {len(cc)}')

            features = e.feature_names_in_

            for f,c in zip(features, cc):
                print(f">>> \tFeature: {f:<30s} Encoded Classes {c}")
            
        if type(e) == MultiLabelBinarizer:
            cc = e.classes_
            print(f'>>> MultiLabelBinarizer {k}: {len(cc)}')
            for c in cc:
                print(f'>>> \tEncoded Classes: \t{k}_{c}')