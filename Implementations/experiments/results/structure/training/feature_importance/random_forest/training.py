import warnings

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

modes = ['RNN_RELU', 'GRU']

without_node_edge = ['layers', 'diameter', 'density', 'average_shortest_path_length', 'eccentricity_mean', 'eccentricity_var', 'eccentricity_std',
                     'degree_mean', 'degree_var', 'degree_std', 'closeness_mean', 'closeness_var', 'closeness_std', 'nodes_betweenness_mean', 'nodes_betweenness_var', 'nodes_betweenness_std',
                     'edge_betweenness_mean', 'edge_betweenness_var', 'edge_betweenness_std']
only_node_edge = ['nodes', 'edges', 'source_nodes', 'sink_nodes']
only_var = ['eccentricity_var', 'degree_var', 'closeness_var', 'nodes_betweenness_var', 'edge_betweenness_var']

features = [without_node_edge, only_node_edge, only_var]
features_title = ['without_node_edge', 'only_node_edge', 'only_var']

scaler = preprocessing.MinMaxScaler()
classifier = RandomForestRegressor()


def write_to_file(file, string, mode):
    f = open('{}.csv'.format(file), f'{mode}')
    f.write(string + '\n')
    f.close()


if __name__ == '__main__':
    write_to_file('r_squared', 'feature_type,RNN_ReLU,GRU', 'w')
    for feature, feature_title in zip(features, features_title):
        write_to_file(f'rf_feature_importance/{feature_title}', 'feature,RNN_ReLU,GRU', 'w')

        importances = {_feature: [] for _feature in feature}
        r_squared = list()

        for mode in modes:
            file = pd.read_csv('../../../{}.csv'.format(mode.lower()))

            X = scaler.fit_transform(file[feature])
            y = file[['test_acc']]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10, random_state=7)

            classifier.fit(X_train, y_train)
            pred = classifier.predict(X_test)
            r_squared.append(r2_score(y_test, pred))
            importance = classifier.feature_importances_

            for i, v in enumerate(importance):
                importances[feature[i]].append(v)

        for key, value in importances.items():
            write_to_file(f'rf_feature_importance/{feature_title}', f'{key},{",".join(str(round(v, 2)) for v in value)}', 'a')
        write_to_file('r_squared', f'{feature_title},{",".join(str(round(r, 2)) for r in r_squared)}', 'a')
