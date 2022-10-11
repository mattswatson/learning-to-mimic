import os
import torch
import torch.nn.functional as F
import imageio
import cv2
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


def main():
    """
    Generate animated version of Figure 4 in the Supplementary Material
    """
    parser = argparse.ArgumentParser(description='Generate animated GIFs of model explanations overtime')

    parser.add_argument('basepath', type=str, default='Path to explanations')
    parser.add_argument('savepath', type=str, default='Path to save results to')

    args = parser.parse_args()

    # Make all image first
    all_paths = os.listdir(args.basepath)

    plt.ioff()
    plt.tight_layout()

    with tqdm(total=len(all_paths)) as progress:
        for p in all_paths:
            fig, axes = plt.subplots(2, 5, figsize=(25, 25))
            axes = axes.flatten()

            epoch = p[5:]

            # Load explanations
            with open(os.path.join(args.basepath, p, 'gradcam.pt'), 'rb') as f:
                gcam = torch.load(f, map_location='cpu')

            # gcam shape: [n_samnples * n_submodels, 1, 1, 7, 7]
            #gcam = gcam[:, 0, :, :]
            gcam = F.interpolate(gcam, size=(224, 224), mode='bilinear', align_corners=False)

            # Normalise
            B, C, H, W = gcam.shape
            gcam = gcam.view(B, -1)
            gcam -= gcam.min(dim=1, keepdim=True)[0]
            gcam /= gcam.max(dim=1, keepdim=True)[0]
            gcam = gcam.view(B, C, H, W)
            gcam = gcam * 255.0

            # Flatten the three channels into one
            gcam = torch.mean(gcam, dim=1)

            # Split the samples into the sub-model groups
            gcam_submodel_images = []
            for submodel in range(10):
                gcam_submodel = gcam[submodel * 209:(submodel + 1) * 209, :, :]
                gcam_submodel = torch.mean(gcam_submodel, dim=0)

                # Only keep the important pixels
                mask = gcam_submodel < gcam_submodel.max() * 0.6
                gcam_submodel[mask] = 0 * gcam_submodel[mask]

                sns.heatmap(gcam_submodel, square=True, cmap=sns.diverging_palette(20, 220, n=200), center=0,
                            ax=axes[submodel], cbar=False)

                # gcam_submodel_images.append(gcam_image)
                # axes[submodel].imshow(gcam_image)
                axes[submodel].set_title('Submodel {}'.format(submodel))

            # Save image
            fig.subplots_adjust(hspace=-0.6)
            plt.savefig(os.path.join(args.savepath, 'epoch{}.png'.format(epoch)), bbox_inches='tight')

            plt.clf()

            progress.update(1)

    # Create a gif based on those images
    with imageio.get_writer(os.path.join(args.savepath, 'animated.gif'), mode='I', duration=1.0) as writer:
        for filename in os.listdir(args.savepath):
            if filename[-3:] != 'png':
                continue

            path = os.path.join(args.savepath, filename)
            image = imageio.imread(path)
            writer.append_data(image)

    print('Done!')


if __name__ == '__main__':
    main()