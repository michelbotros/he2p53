import matplotlib.pyplot as plt
import numpy as np


def plot_pred_batch(x, y, y_hat, save_path=None, patches=3):
    """ Plots the p53 prediction. Standard 3x3 grid (HE, P53, predicted p53)
    The plot is stored in the experiment dir to keep track of performance.

    Args:
        x: HE batch [B, H, W, 3]
        y: true p53 batch [B, H, W, 3]
        y_hat: predicted p53 batch [B, H, W, 3]
        save_path: path where plot of predictions is stored.
        patches: how many patches to include in the plot

    Returns:
        none: saves the figure at the save patch
    """
    patches = min(len(x), patches)

    # show just the image
    fig, axes = plt.subplots(3, patches, figsize=(20, 20), squeeze=False)

    data = np.stack((x, y, y_hat), axis=1)[:patches]

    for row in range(patches):
        for col in range(3):
            if row == 0:
                axes[row, 0].title.set_text('HE')
                axes[row, 1].title.set_text('p53')
                axes[row, 2].title.set_text(r'$\hat{p53}$')
            axes[row, col].imshow(data[row, col])

    for ax in axes.flatten():
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
