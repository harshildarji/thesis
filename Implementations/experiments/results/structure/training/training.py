import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

modes = ["RNN_TANH", "RNN_RELU", "GRU", "LSTM"]
features = [
    "layers",
    "nodes",
    "edges",
    "source_nodes",
    "sink_nodes",
    "diameter",
    "density",
    "average_shortest_path_length",
    "eccentricity_mean",
    "eccentricity_var",
    "eccentricity_std",
    "degree_mean",
    "degree_var",
    "degree_std",
    "closeness_mean",
    "closeness_var",
    "closeness_std",
    "nodes_betweenness_mean",
    "nodes_betweenness_var",
    "nodes_betweenness_std",
    "edge_betweenness_mean",
    "edge_betweenness_var",
    "edge_betweenness_std",
]

scaler = preprocessing.MinMaxScaler()
classifier_names = ["BayesianRidge", "RandomForestRegressor", "AdaBoostRegressor"]
classifiers = [BayesianRidge(), RandomForestRegressor(), AdaBoostRegressor()]


def write_to_file(file, string, mode):
    f = open("{}.csv".format(file), f"{mode}")
    f.write(string + "\n")
    f.close()


if __name__ == "__main__":
    write_to_file(
        "r_squared", "mode,BayesianRidge,RandomForestRegressor,AdaBoostRegressor", "w"
    )

    for mode in modes:
        write_to_file(
            f"feature_importance/{mode.lower()}",
            "feature,BayesianRidge,RandomForestRegressor,AdaBoostRegressor",
            "w",
        )
        file = pd.read_csv("../{}.csv".format(mode.lower()))

        X = scaler.fit_transform(file[features])
        y = file[["test_acc"]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=7
        )

        r_squared = list()
        importances = {feature: [] for feature in features}
        for clf_name, clf in zip(classifier_names, classifiers):
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            r_squared.append(r2_score(y_test, pred))

            try:
                importance = clf.feature_importances_
            except:
                importance = clf.coef_
                if isinstance(importance[0], np.ndarray):
                    importance = importance[0]

            for i, v in enumerate(importance):
                importances[features[i]].append(v)

        for key, value in importances.items():
            write_to_file(
                f"feature_importance/{mode.lower()}",
                f'{key},{",".join(str(v) for v in value)}',
                "a",
            )
        write_to_file("r_squared", f'{mode},{",".join(str(r) for r in r_squared)}', "a")
