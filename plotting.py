import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import seaborn as sns
import collections
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_hex


rng = np.random.default_rng(0)


colors = ['xkcd:sky', 'xkcd:grass']
cmap = ListedColormap(colors)


def draw_tree(clf, colors=colors, **kwargs):
    import pydotplus

    d = tree.export_graphviz(clf, out_file=None, filled=True, **kwargs)
    graph = pydotplus.graph_from_dot_data(d)

    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(to_hex(colors[i]))

    graph.set_dpi(300)
    return graph.create(format="png")


def plot_bars_and_confusion(
    truth,
    prediction,
    axes=None,
    vmin=None,
    vmax=None,
    cmap='inferno',
    title=None,
    bar_color=None,
):
    accuracy = accuracy_score(truth, prediction)
    cm = confusion_matrix(truth, prediction)

    if not isinstance(truth, pd.Series):
        truth = pd.Series(truth)

    if not isinstance(prediction, pd.Series):
        prediction = pd.Series(prediction)

    correct = pd.Series(np.where(truth.values == prediction.values, 'Korrekt', 'Falsch'))

    # truth.sort_index(inplace=True)
    # prediction.sort_index(inplace=True)

    if not axes:
        fig, axes = plt.subplots(1, 2)

    if not vmin:
        vmin = cm.min()

    if not vmax:
        vmax = cm.max()

    if not bar_color:
        correct.value_counts().plot.barh(ax=axes[0])
    else:
        correct.value_counts().plot.barh(ax=axes[0], color=bar_color)

    axes[0].text(150, 0.5, "Accuracy {:0.3f}".format(accuracy))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=["Gestorben", "Überlebt"],
        yticklabels=["Gestorben", "Überlebt"],
        ax=axes[1],
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_xlabel("Vorhersage")
    axes[1].set_ylabel("Wahrheit")
    axes[1].set_aspect(1)
    if title:
        plt.suptitle(title)
