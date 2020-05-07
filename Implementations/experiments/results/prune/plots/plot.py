import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.figsize': (15, 8)})
plt.rcParams.update({'legend.loc': 'best', 'legend.framealpha': 0.5})

plt.ylim(0.5, 1.0)

MODES = ['RNN_TANH', 'RNN_RELU', 'GRU', 'LSTM']
PERCENT = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def plot_prune(group, title, regain_title):
    prune = data_mode.get_group(group).groupby(['prune'])
    prune_test_acc = [float(prune.get_group(percent)['test_acc'].iloc[0]) for percent in PERCENT]
    epochs = prune.get_group(10)['epoch'].astype(int).tolist()

    plt.title('{} {}'.format(mode, title))
    plt.xlabel('% Pruned')
    plt.xticks(PERCENT)
    plt.ylabel('Accuracy')
    plt.plot(PERCENT, prune_test_acc, label='Test')
    plt.legend()
    plt.savefig('{}/{}_{}.png'.format(mode, mode.lower(), title.lower().replace(' ', '_')))
    plt.close()

    plt.title('{} {}'.format(mode, regain_title))
    plt.xlabel('Epochs')
    plt.xticks(epochs)
    plt.ylabel('Accuracy')
    for percent in PERCENT:
        percent_pruned_acc = prune.get_group(percent)['test_acc'].astype(float).tolist()
        threshold = .92 if 'tanh' in group else .95
        if percent_pruned_acc[0] > threshold:
            continue
        plt.plot(epochs, percent_pruned_acc, label='Pruned {}%'.format(percent))
    plt.legend()
    plt.savefig('{}/{}_{}.png'.format(mode, mode.lower(), regain_title.lower().replace(' ', '_')))
    plt.close()


for mode in MODES:
    data = pd.read_csv('../{}.csv'.format(mode.lower()))

    data_mode = data.groupby(['mode'])

    # --- plot train-test accuracy of first 50 epochs ---

    train_acc = data_mode.get_group(mode.lower())['train_acc'].astype(float).tolist()
    test_acc = data_mode.get_group(mode.lower())['test_acc'].astype(float).tolist()
    epochs = data_mode.get_group(mode.lower())['epoch'].astype(float).tolist()

    plt.title('{} Accuracy'.format(mode))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_acc, label='Train')
    plt.plot(epochs, test_acc, label='Test')
    plt.legend()
    plt.savefig('{}/{}_accuracy.png'.format(mode, mode.lower()))
    plt.close()

    # --- plot pruning evaluation and required epochs to regain the accuracy ---

    plot_prune('{}_prune'.format(mode.lower()), 'Pruning Evaluation', 'Accuracy Regain')
    plot_prune('{}_prune_i2h'.format(mode.lower()), 'I2H Pruning Evaluation', 'I2H Accuracy Regain')
    plot_prune('{}_prune_h2h'.format(mode.lower()), 'H2H Pruning Evaluation', 'H2H Accuracy Regain')
