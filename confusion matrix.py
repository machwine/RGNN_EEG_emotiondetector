import itertools
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


# trans_prob_mat = (trans_mat.T / np.sum(trans_mat, 1)).T
trans_prob_mat = np.mat('81.25, 13.66, 5.09;15.14, 83.71, 1.15;3.99, 3.46, 92.55')

#trans_prob_mat = [[89.66, 5.38, 4.96][0.9, 96.33, 2.77][0.44, 4.9, 94.66]]

if True:
    #label = ["Patt {}".format(i) for i in range(1, trans_prob_mat.shape[0] + 1)]
    xtick = ['negative', 'neutral', 'positive']
    ytick = ['negative', 'neutral', 'positive']

    #df = pd.DataFrame(trans_prob_mat, index=label, columns=label)

    print(trans_prob_mat)
    #print(df)

    # Plot
    plt.figure(figsize=(7.5, 6.3))
    ax = sns.heatmap(trans_prob_mat, fmt='g', xticklabels=xtick,
                     yticklabels=ytick, cmap='Blues',
                     linewidths=6, annot=True, cbar= False)

    # Decorations
    plt.xticks(fontsize=16, family='Times New Roman')
    plt.yticks(fontsize=16, family='Times New Roman')

    plt.tight_layout()

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.show()
