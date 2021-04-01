import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

plt.rcParams.update({'figure.figsize': (15, 8)})
plt.rcParams.update({'legend.loc': 'best', 'legend.framealpha': 0.5})

modes = ['RNN_TANH', 'RNN_RELU', 'GRU', 'LSTM']

for mode in modes:
    file = pd.read_csv('../{}.csv'.format(mode.lower()))

    # --- plot correlation between test_acc and other graph properties ---

    plt.rcParams.update({'font.size': 10})

    f, ax = plt.subplots(figsize=(10, 8))
    corr = file.drop(['graph_nr'], axis=1).corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap='coolwarm', square=True, ax=ax, linewidths=.2)
    plt.title('{} - Correlations between different graph properties and test results'.format(mode))
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.savefig('{}/corr_with_test_acc.png'.format(mode))
    plt.close()

    # --- plot jointplots for test_acc and a few graph properties ---

    plt.rcParams.update({'font.size': 24})

    cols = ['layers', 'nodes', 'edges', 'source_nodes', 'diameter', 'density', 'average_shortest_path_length', 'eccentricity_var', 'degree_var', 'closeness_var', 'nodes_betweenness_var', 'edge_betweenness_var']
    for c in cols:
        pearson, p = scipy.stats.pearsonr(file['test_acc'], (file[c]))
        sns.jointplot(x=file['test_acc'], y=file[c], kind='reg', height=10, xlim=(file['test_acc'].min() - .04, 1.0)).plot_joint(sns.kdeplot, zorder=0, n_levels=5, alpha=.2, color='k', shade=False)
        plt.figtext(0.32, 0, 'pearsonr = {:.2f}; p = {:.2f}'.format(pearson, p))
        plt.suptitle('Jointplot for test_acc and {}'.format(c), y=1.01)
        plt.savefig('{}/test_acc_jointplots/jointplot_test_acc_{}.png'.format(mode, c), bbox_inches='tight', pad_inches=.1)
        plt.close()

    # --- plot joinplot for nodes and layers ---

    sns.jointplot(x=file['nodes'], y=file['layers'], xlim=(5, 55), ylim=(1, file['layers'].max() + 2), height=10, marginal_kws=dict(bins=10, rug=True), annot_kws=dict(stat='r'), s=40, edgecolor='w', linewidth=1).plot_joint(sns.kdeplot, zorder=0, n_levels=5, alpha=.2, color='k', shade=True)
    plt.suptitle('Jointplot for nodes and layers'.format(c), y=1.01)
    plt.savefig('{}/jointplot_nodes_layers.png'.format(mode), bbox_inches='tight', pad_inches=.1)
    plt.close()

    # --- plot catplot for different graph properties and graph type to compare test_acc? ---

    plt.rcParams.update({'font.size': 15})

    Y = ['test_acc', 'layers', 'nodes', 'edges', 'density']
    for i, y in enumerate(Y):
        sns.catplot(x='graph', y=y, data=file)
        plt.savefig('{}/graph_properties/graph_{}'.format(mode, y))
        plt.close()
