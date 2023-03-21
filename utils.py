import matplotlib.pyplot as plt
import numpy as np
from dataset import DAB_density_norm
from skimage.color import rgb2hed, hed2rgb


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


def plot_hed_space(he_rgb, ihc_rgb, dab_true, dab_pred, patches=1, save_path=None):

    # convert to HED space
    ihc_hed = rgb2hed(ihc_rgb)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    # Display
    fig, axes = plt.subplots(patches, 3, figsize=(15, 6), squeeze=False)
    ax = axes.ravel()

    ax[0].imshow(he_rgb)
    ax[0].set_title("HE")
    ax[1].imshow(ihc_rgb)
    ax[1].set_title("p53")
    ax[2].imshow(ihc_d)
    ax[2].set_title("DAB from p53\nDAB density: {:.4f}\npredicted: {:.4f} ".format(dab_true, dab_pred))
    for a in ax.ravel():
        a.axis('off')

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()