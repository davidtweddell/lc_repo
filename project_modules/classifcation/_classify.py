# Some of these packages are no longer being used, but I haven't cleared them out yet.
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, make_scorer,roc_curve, roc_auc_score,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from copy import deepcopy

from multiprocessing import Pool, cpu_count
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from boruta import BorutaPy
import csv


def getXY(df, target_col="LC_STATUS"):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def classify_MP(
    X,
    y,
    lCV=[3],
    lTitle=None,
    times=5,
    lClassifiers=["RF", "ET", "DT", "RFR", "ETR", "LinearSVC", "LogR"],
    lTrees=[1, 10],
    lMD=[None, 10],
    randomstate=42,
    n_jobs=-1,
    lScorers=["accuracy", "roc_auc", "neg_mean_squared_error"],
    filename=None,
    verbose=False,
):

    treeParams = {
        "clf__max_depth": lMD,
        "clf__random_state": [randomstate],
    }

    # https://scikit-learn.org/stable/modules/compose.html -> nested params -> clf=classifier bc step in pipeline which should be ('string',estimator()) -> in our case string=clf -> thus clf in the dict becomes the classifier specified and clf__parameter becomes the parameter for the clf
    dictNonScaleClfs = {
        "RF": {
            "clf": [RandomForestClassifier()],
            "clf__n_estimators": lTrees,
            **treeParams,
        },
        "RFR": {
            "clf": [RandomForestRegressor()],
            "clf__n_estimators": lTrees,
            **treeParams,
        },
        "ET": {
            "clf": [ExtraTreesClassifier()],
            "clf__n_estimators": lTrees,
            **treeParams,
        },
        "ETR": {
            "clf": [ExtraTreesRegressor()],
            "clf__n_estimators": lTrees,
            **treeParams,
        },
        "DT": {"clf": [DecisionTreeClassifier()], **treeParams},
        "LogR": {"clf": [LogisticRegression()], "clf__max_iter": [1000]},
    }

    dictScaleClfs = {
        "LinearSVC": {
            "clf": [LinearSVC()],
        }
    }

    lOut = []
    # for m, X in enumerate(lMat):
    #     y = lLabel[m]
    results = []

    if len(set(y)) > 2 and "roc_auc" in lScorers:
        lScoresSel = deepcopy(lScorers)
        lScoresSel.remove("roc_auc")
    else:
        lScoresSel = lScorers

    dictScorer = {}
    for i,score in enumerate(lScoresSel):
        if score == 'sensitivity':
            dictScorer[score] = make_scorer(__sensitivity_value)
        elif score == "specificity":
            dictScorer[score] = make_scorer(__specificity_value)
        elif score == 'NPV':
            dictScorer[score] = make_scorer(__negative_predictive_value)
        elif score == 'PPV':
            dictScorer[score] = make_scorer(__positive_predictive_value)
        else:
            dictScorer[score] = score

    for c in lCV:
        pipe = Pipeline(
            steps=[("clf", None)]
        )  # https://scikit-learn.org/stable/modules/compose.html -> nested params

        params = list(filter(None, [*map(dictNonScaleClfs.get, lClassifiers)]))
        paramsScale = list(
            filter(None, [*map(dictScaleClfs.get, lClassifiers)])
        )

        RTmp = []
        if len(params) != 0:
            clf = GridSearchCV(
                pipe,
                param_grid=params,
                scoring=dictScorer,
                refit="accuracy",
                n_jobs=n_jobs,
                cv=StratifiedKFold(c),
            )

            clf.fit(X, y)
            RTmp.append(clf.cv_results_)

        if len(paramsScale) != 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = GridSearchCV(
                pipe,
                param_grid=paramsScale,
                scoring=dictScorer,
                refit="accuracy",
                n_jobs=n_jobs,
                cv=StratifiedKFold(c),
            )

            clf.fit(X_scaled, y)
            RTmp.append(clf.cv_results_)

        for R in RTmp:
            dR = pd.DataFrame(R)
            dR["cv"] = c
            # dR["title"] = lTitle[m]
            results.append(dR)

        dResults = pd.concat(results)

        # uniqueParams = dResults.params.drop_duplicates()
        # uniqueParams.shape

        # lMeans = []
        # for param in uniqueParams:
        #     paramResults = dResults[dResults["params"] == param]
        #     for c in lCV:
        #         for t in lTitle:
        #             dfTmp = paramResults.loc[
        #                 (paramResults["params"] == param)
        #                 & (paramResults["title"] == t)
        #                 & (paramResults["cv"] == c)
        #             ]
        #             dfMean = dfTmp.mean()
        #             dfMean["params"] = dfTmp["params"].iloc[0].__str__()
        #             dfMean["param_clf"] = dfTmp["param_clf"].iloc[0].__str__()
        #             dfMean["title"] = dfTmp["title"].iloc[0]

        #             clf = dfTmp["param_clf"].iloc[0].__str__()
        #             clfInd = clf.index("(")
        #             clf = clf[:clfInd]
        #             dfMean["Classifier"] = clf

        #             lMeans.append(dfMean)

        # dMeansComb = pd.concat(lMeans, axis=1).T
        # dOut = __summarizeClassifyResultsMeans(dMeansComb)
        # lOut.append(dOut)
        # if filename is not None:
        #     dResults.to_csv(f"{filename}-{lTitle[m]}.csv", index=False)
        #     dMeansComb.to_csv(f"{filename}-MEANS-{lTitle[m]}.csv", index=False)
        #     dOut.to_csv(f"{filename}-SUMMARY-{lTitle[m]}.csv", index=False)

    return dResults


def boruta_fs(
    X,
    y,
    feat_list,
    trees="auto",
    ittr=100,
    threshold=100,
    top_rank=5,
    model=RandomForestClassifier(n_jobs=-1, class_weight="balanced"),
    fileName=None,
    verbose=0,
    random_state=42
):
    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels

    # define Boruta feature selection method
    feat_selector = BorutaPy(
        model,
        n_estimators=trees,
        max_iter=ittr,
        perc=threshold,
        verbose=verbose,
        random_state=random_state,
    )  # , verbose=1

    # find all relevant features
    feat_selector.fit(X, y)

    # check selected features
    sup = feat_selector.support_

    # check ranking of features
    rank = feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    true_feat = []
    top_rank_feat = []
    for i in range(len(feat_selector.support_)):
        if feat_selector.ranking_[i] <= top_rank:
            print(
                f"{feat_selector.ranking_[i]}\t{feat_selector.support_[i]}\t{feat_list[i]}"
            )
            top_rank_feat.append(feat_list[i])
            if feat_selector.support_[i] == True:
                true_feat.append(feat_list[i])
    print(true_feat)

    if fileName != None:
        with open(fileName, "w") as file:
            csvwrite = csv.writer(
                file,
            )
            for i in range(len(feat_selector.support_)):
                if feat_selector.ranking_[i] <= top_rank:
                    print(
                        f"{feat_selector.ranking_[i]}\t{feat_selector.support_[i]}\t{feat_list[i]}"
                    )
                    csvwrite.writerow(
                        [
                            feat_selector.ranking_[i],
                            feat_selector.support_[i],
                            feat_list[i],
                        ]
                    )
    return true_feat, top_rank_feat


def boruta_MP(
    X,
    y,
    l_features=None,
    l_title=None,
    nest="auto",
    nittr=200,
    threshold=100,
    top_rank=5,
    fileName=None,
    model=RandomForestClassifier(n_jobs=-1, class_weight="balanced"),
):
    print(f"---___---Running Boruta---___----")
    
    l_selected_feat = []
    pool_itter_args = []
    
    pool_itter_args.append(
        (X, y, l_features, nest, nittr, threshold, top_rank, model)
    )

    pool = Pool()
    l_selected_feat = pool.starmap(boruta_fs, pool_itter_args)
    pool.close()
    pool.join()

    true_feat = []
    top_rank_feat = []
    for i in l_selected_feat:
        true_feat.append(i[0])
        top_rank_feat.append(i[1])
    # print (f"----SELECTED FEATURES BELOW---")
    # for i in range(len(l_mat)):
    #     print(f"{l_title[i]}--Selected Feat: {l_selected_feat[i]}")

    if fileName != None:
        for i, title in enumerate(l_title):

            ltrue = ["True"] * len(true_feat[i])
            ltop = ["Top_Rank"] * len(top_rank_feat[i])
            outTrue = np.vstack([ltrue, true_feat[i]])
            outTop = np.vstack([ltop, top_rank_feat[i]])

            outTrue = pd.DataFrame(outTrue.T, columns=["Type", "Feature"])
            outTop = pd.DataFrame(outTop.T, columns=["Type", "Feature"])

            print(outTrue)
            print(outTop)
            outFile = outTrue.append(outTop)
            # outFile.to_excel(f'{fileName}_{title.replace(" ", "_")}_T{nest}_ittr{nittr}_thres{threshold}_topRank{top_rank}.xlsx',index=False)
            outFile.to_csv(
                f'{fileName}_{title.replace(" ", "_")}_T{nest}_ittr{nittr}_thres{threshold}_topRank{top_rank}.csv',
                index=False,
            )

    return true_feat, top_rank_feat


def __summarizeClassifyResultsMeans(dfMeans, bPrint=False):
    dfMeans = deepcopy(dfMeans)
    lMeans = [
        x
        for x in dfMeans.columns.to_list()
        if ("mean_" in x[:10]) and ("time" not in x)
    ]
    lParams = [x for x in dfMeans.columns.to_list() if ("param_clf" in x[:10])]

    lOutput = ["title", "Classifier", "cv"] + lParams + lMeans + ["params"]
    dfOut = dfMeans[lOutput]
    if bPrint:
        print(dfOut.to_markdown())
    return dfOut


def __positive_predictive_value(
    y_true, y_pred
):  # NOTE: - this is the exaclty same results as "precision" in sklearn
    # Calculate the confusion matrix
    # print(y_true,y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate PPV (Positive Predictive Value)
    try:
        # print(tn,fp,fn,tp)
        if (
            tp + fp == 0
        ):  # NOTE - this is an edgecase -> based on https://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
            ppv = 0.0
        else:
            ppv = tp / (tp + fp)
        return ppv
    except ZeroDivisionError:
        return 0.0  # Return a predefined value when PPV is ill-defined
    except RuntimeWarning as rw:
        print(f"{tn}-{fp}-{fn}-{tp}")
        return np.nan


def __negative_predictive_value(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate NPV (Negative Predictive Value)
    try:
        if (
            tn + fn == 0
        ):  # NOTE - this is an edgecase -> based on https://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
            ppv = 0.0
            npv = 0.0
        else:
            npv = tn / (tn + fn)
        return npv
    except ZeroDivisionError:
        return 0.0  # Return a predefined value when NPV is ill-defined
    except RuntimeWarning:
        print(f"{tn}-{fp}-{fn}-{tp}")
        return np.nan


def __sensitivity_value(
    y_true, y_pred
):  # NOTE: - this is the exaclty same results as "recall" in sklearn
    from sklearn.metrics import confusion_matrix

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate Sensitivity == TPR (True Positive Rate) == Recall
    try:
        if (
            tp + fn == 0
        ):  # NOTE - this is an edgecase -> based on https://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
            sens = 0.0
        else:
            sens = tp / (tp + fn)
        return sens
    except ZeroDivisionError:
        return 0.0  # Return a predefined value when NPV is ill-defined
    except RuntimeWarning:
        print(f"{tn}-{fp}-{fn}-{tp}")
        return np.nan


def __specificity_value(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate Sepcificity == TNR (True Negative Rate)
    try:
        if (
            tn + fp == 0
        ):  # NOTE - this is an edgecase -> based on https://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases
            spec = 0.0
        else:
            spec = tn / (tn + fp)
        return spec
    except ZeroDivisionError:
        return 0.0  # Return a predefined value when NPV is ill-defined
    except RuntimeWarning:
        print(f"{tn}-{fp}-{fn}-{tp}")
        return np.nan
