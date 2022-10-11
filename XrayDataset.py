import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image

import os

from utils import create_heatmap, load_image_numpy


class GazeXrayDataset(Dataset):
    def __init__(self, gaze_path, cxr_jpg_path, cxr_transforms=None, heatmap_transforms=None, heatmap_threshold=None,
                 sigma=150, generated_heatmaps_path=None, return_binary_heatmap=False, repeat=None):
        """
        :param gaze_path: Path to base EGD dataset
        :param cxr_jpg_path: Path to the original CXR images in JPEG format
        :param cxr_transforms: A list of torchvision.transforms to perform on the CXRs
        :param heatmap_threshold: Only include heatmap pixels larger than this threshold
        :param sigma: Integer value to use for standard deviation of the Gaussian used when generating heatmaps
        :param generated_heatmaps_path: Path to pre-generated heatmaps. If None, new heatmaps are generated from raw
        fixations
        :param repeat: (int), number of time to repeat the dataset
        """
        super(GazeXrayDataset, self).__init__()

        self.gaze_path = gaze_path
        self.cxr_jpg_path = cxr_jpg_path
        self.cxr_transforms = cxr_transforms
        self.heatmap_transforms = heatmap_transforms
        self.heatmap_threshold = heatmap_threshold
        self.sigma = sigma
        self.generated_heatmaps_path = generated_heatmaps_path
        self.return_binary_heatmap = return_binary_heatmap
        self.repeat = repeat

        # These are the three labels we're interested in
        self.class_names = ['Normal', 'CHF', 'pneumonia']

        # CSV file that contains case information
        self.master_sheet = pd.read_csv(os.path.join(self.gaze_path, 'master_sheet.csv'))

        # CSV file with fixations - we only need this if we're not using generated heatmaps
        if not self.generated_heatmaps_path:
            self.fixations = pd.read_csv(os.path.join(self.gaze_path, 'fixations.csv'))
        else:
            # If we're using generated heatmaps, use only CXRs we also have the heatmaps for
            dicoms_with_heatmaps = [p for p in os.listdir(self.generated_heatmaps_path) if
                                    os.path.isdir(os.path.join(self.generated_heatmaps_path, p))]

            self.master_sheet = self.master_sheet[self.master_sheet['dicom_id'].isin(dicoms_with_heatmaps)]

        self.dicom_ids = list(self.master_sheet['dicom_id'].unique())

        if self.repeat:
            self.dicom_ids *= self.repeat

    def __len__(self):
        return len(self.dicom_ids)

    def get_heatmap(self, dicom_id, return_binary_heatmap=False):
        """
        Returns either a pre-generated or newly-generated heatmap for the given DICOM ID, depending on class
        variables
        :param dicom_id: ID of the CXR image to generate the heatmap for
        :return: 1D numpy array
        """

        # If we're given a generated_heatmaps_path then we just want to load the generated heatmap
        if self.generated_heatmaps_path:
            heatmap_path = os.path.join(self.generated_heatmaps_path, dicom_id, 'heatmap.png')

            try:
                heatmap = load_image_numpy(heatmap_path)
            except FileNotFoundError:
                print('Heatmap for CXR {} not found at {}! Heatmap will be all zeros.'.format(dicom_id, heatmap_path))

            # We don't care about the alpha channel
            heatmap = heatmap[:, :, :3]
        else:
            # Otherwise we need to generate it from scratch
            heatmap = create_heatmap(dicom_id, self.fixations, self.master_sheet, self.cxr_jpg_path, sigma=self.sigma)

        # Torch expects images to be of shape (C, H, W)
        if len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
            heatmap = np.transpose(heatmap, (2, 0, 1))

        if len(heatmap.shape) != 3:
            heatmap = np.stack([heatmap, heatmap, heatmap], axis=0)

        # If we're returning a binary heatmap, we should just return the fixation locations as a binary image
        if return_binary_heatmap:
            fixations_mask = np.zeros_like(heatmap)[:, :, 0]

            cxr_fixations = self.fixations[self.fixations['DICOM_ID'] == dicom_id]

            for i, row in cxr_fixations.iterrows():
                x_coord = row['X_ORIGINAL']
                y_coord = row['Y_ORIGINAL']

                size = 10

                # Make a 10x10 box
                fixations_mask[y_coord - size:y_coord + size, x_coord - size:x_coord + size] = 255

            return heatmap, fixations_mask

        return heatmap

    def __getitem__(self, item):
        """
        :param item: ID (not DICOM ID!) to return
        :return cxr: numpy array of the CXR image
        :return heatmap: numpy array of the heatmap
        :return labels: numpy array of a one-hot encoding of the labels ['Normal', 'CHF', 'pneumonia']
        """
        # This is the DICOM ID we want to get
        dicom_id = self.dicom_ids[item]

        # Let's get a one-hot encoded label: [normal, chf, pneumothorax]
        study_info = self.master_sheet[self.master_sheet['dicom_id'] == dicom_id]
        labels = np.array(study_info[self.class_names], dtype='int32')

        cxr_file_path = study_info['path'].values[0]
        cxr_file_path = os.path.join(self.cxr_jpg_path, cxr_file_path[:-4] + '.jpg')
        cxr = load_image_numpy(cxr_file_path).astype('float32')

        if self.return_binary_heatmap:
            heatmap, binary_heatmap = self.get_heatmap(dicom_id, self.return_binary_heatmap)
            heatmap = heatmap.astype('float32')
        else:
            heatmap = self.get_heatmap(dicom_id).astype('float32')

        # If we've loaded in a pre-generated heatmap, our CXR will need resizing
        if cxr.shape != heatmap.shape[1:]:
            if not self.generated_heatmaps_path:
                # If we haven't loaded in a pre-generated heatmap but we still have the wrong shape, something has
                # gone wrong
                raise Exception('CXR had shape {} and heatmap had shape {}: if generating heatmaps, these should be'
                                'the same size!'.format(cxr.shape, heatmap.shape))

            # Otherwise all we need to to is resize the CXR (last shape element is channel dimension)
            cxr = cv2.resize(cxr, (heatmap.shape[2], heatmap.shape[1]))

        # We probably want colour images
        if len(cxr.shape) != 3:
            cxr = np.stack([cxr, cxr, cxr], axis=0)

        cxr = torch.tensor(cxr)
        heatmap = torch.tensor(heatmap)

        # We want images in the range [0, 1] not [0, 255]
        if torch.max(cxr) > 1:
            cxr = cxr / 255

        if torch.max(heatmap) > 1:
            heatmap = heatmap / 255

        if self.cxr_transforms:
            cxr_transform = transforms.Compose(self.cxr_transforms)
            cxr = cxr_transform(cxr)

        if self.heatmap_transforms:
            heatmap_transform = transforms.Compose(self.heatmap_transforms)
            heatmap = heatmap_transform(heatmap)

        if self.return_binary_heatmap:
            return cxr, heatmap, binary_heatmap, labels

        return cxr, heatmap, labels
