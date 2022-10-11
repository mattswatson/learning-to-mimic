import pandas as pd
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from visdom import Visdom
import torch
import cv2
import torch.nn as nn
import matplotlib.cm as cm
import torch.nn.functional as F
from textwrap import wrap
from tqdm import tqdm as tqdm_write
from collections import OrderedDict

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
import pickle

from math import exp

from PIL import Image


def load_image_numpy(path):
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype='int32')

    # Convert to a (height, width, channel) shape
    # data = data[:, :, None]
    return data


def crop(image):
    '''
    Auxilary function to crop image to non-zero area
    :param image: input image
    :return: cropped image
    '''
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def create_heatmap(dicom_id, fixations, master_sheet, cxr_jpg_path, save_heatmap=None, sigma=150):
    """
    Generate a grayscale heatmap based on fixation points

    :param dicom_id: the DICOM ID for the CXR
    :param fixations: Pandas DF of all fixations
    :param master_sheet: Pandas DF of studies
    :param save_heatmap: If not None, directory to save heatmap to
    :param sigma: Standard deviation for the Gaussian filter. Sigma=150 is used in the original dataset paper
    :return: Grayscale heatmap for given CXR image
    """
    study_info = master_sheet[master_sheet['dicom_id'] == dicom_id]

    # Open the image and show it
    cxr_file_path = study_info['path'].values[0]

    # We're using JPEGs not DICOMS, so change extension
    cxr_file_path = os.path.join(cxr_jpg_path, cxr_file_path[:-4] + '.jpg')

    cxr = load_image_numpy(cxr_file_path)

    # Plot the fixations on the image
    cxr_fixations = fixations[fixations['DICOM_ID'] == dicom_id]
    fixations_mask = np.zeros_like(cxr)

    for i, row in cxr_fixations.iterrows():
        x_coord = row['X_ORIGINAL']
        y_coord = row['Y_ORIGINAL']

        size = 10

        # Make a 10x10 box
        fixations_mask[y_coord - size:y_coord + size, x_coord - size:x_coord + size] = 255

    heatmap = ndimage.gaussian_filter(fixations_mask, sigma)

    if save_heatmap is not None:
        plt.imsave(os.path.join(save_heatmap, '{}-heatmap.jpg'.format(dicom_id)), heatmap)

    # Try scaling the heatmap back to the range [0, 255]
    heatmap = ((heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))) * 255

    return heatmap


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomLinePlotter(object):
    def __init__(self, env_name='main', server='localhost', port=8097):
        self.vis = Visdom(server=server, port=port)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y, xlabel='Epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.vis.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')

    def plot_matplotlib(self, plot_name, plt):
        self.plots[plot_name] = self.vis.matplot(plt,env=self.env)

    def plot_text(self, text, title='Text'):
        self.vis.text(text, env=self.env, opts=dict(title=title))


# -------------------- GradCAM ------------------

# -- Code modified from source: https://github.com/kazuto1011/grad-cam-pytorch
class _BaseWrapper(object):
    """
    Please modify forward() and backward() depending on your task.
    """
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def generate(self):
        raise NotImplementedError

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return list(zip(*self.probs.sort(0, True)))  # element: (probability, index)


class GradCam(_BaseWrapper):
    def __init__(self, model, candidate_layers=[]):
        super(GradCam, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers

        def forward_hook(module, input, output):
            self.fmap_pool[id(module)] = output.detach()


        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            if len(self.candidate_layers) == 0 or module[0] in self.candidate_layers:
                self.handlers.append(module[1].register_forward_hook(forward_hook))
                self.handlers.append(module[1].register_backward_hook(backward_hook))

    def find(self, pool, target_layer):
        # --- Query the right layer and return it's value.
        for key, value in pool.items():
            for module in self.model.named_modules():
                # print(module[0], id(module[1]), key)
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError(f"Invalid Layer Name: {target_layer}")

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads ,2))) + 1e-5
        return grads /l2_norm

    def compute_grad_weights(self, grads):
        grads = self.normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)


    def generate(self, target_layer):
        fmaps = self.find(self.fmap_pool, target_layer)
        grads = self.find(self.grad_pool, target_layer)
        weights = self.compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.0)

        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam


def compute_gradCAM(probs, labels, gcam, testing_labels, criterion, target_layer='encoder.blocks.6'):
    # --- one hot encode this:
    # one_hot = torch.zeros((labels.shape[0], labels.shape[1])).float()
    one_hot = torch.zeros((probs.shape[0], probs.shape[1])).float()
    max_int = torch.max(criterion(probs), 1)[1]

    if testing_labels:
        for i in range(one_hot.shape[0]):
            one_hot[i][max_int[i]] = 1.0

    else:
        for i in range(one_hot.shape[0]):
            one_hot[i][torch.max(labels, 1)[1][i]] = 1.0

    probs.backward(gradient=one_hot.cuda(), retain_graph=True)
    fmaps = gcam.find(gcam.fmap_pool, target_layer)
    grads = gcam.find(gcam.grad_pool, target_layer)

    weights = F.adaptive_avg_pool2d(grads, 1)
    gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    gcam_out = F.relu(gcam)
    return probs, gcam_out, one_hot

# -------- End Grad-CAM code

def train_binary_svm(neg_class, pos_class, feature_names=None, test_split=0.3, kernel=None, save=None,
                     save_results=None, results_name='', scale=True):
    # Create a pandas dataframe for ease
    # In theory these should have the same length, but it's good to just check
    num_features = max([len(l) for l in neg_class] + [len(l) for l in pos_class])

    if feature_names is None:
        columns = [str(i) for i in range(num_features)]
    else:
        columns = feature_names

    df = pd.DataFrame(neg_class + pos_class, columns=columns)
    labels = pd.DataFrame.from_records([[0] for _ in range(len(neg_class))] +
                                       [[1] for _ in range(len(pos_class))])

    df = df.fillna(0)

    # Split the data up, normalise it
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split)
    split = sss.split(df, labels)

    for train_indices, test_indices in split:
        data_train, labels_train = df.iloc[train_indices], labels.iloc[train_indices]
        data_test, labels_test = df.iloc[test_indices], labels.iloc[test_indices]

    data_train = preprocessing.scale(data_train)
    data_test = preprocessing.scale(data_test)

    if scale:
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

    # Train all SVMs, unless we're asked for a specific one
    possible_kernels = ['linear', 'poly', 'rbf', 'sigmoid'] if kernel is None else [kernel]

    for k in possible_kernels:
        print("======================= Training {} SVM".format(k))

        svm = SVC(kernel=k, verbose=True, shrinking=True, cache_size=500)
        svm.fit(data_train, labels_train)

        if save is not None:
            filename = k + '-svm.pkl'
            path = os.path.join(save, filename)

            print("======================= Saving {} SVM to {}".format(k, path))

            with open(path, 'wb') as f:
                pickle.dump(svm, f)

        print("======================= Evaluating SVM")

        preds = svm.predict(data_test)

        conf_matrix = confusion_matrix(labels_test, preds).tolist()
        class_report = classification_report(labels_test, preds)
        results = "{} \n\n {}".format(conf_matrix, class_report)

        print(results)

        if save_results is not None:
            filename = '{}-results{}.txt'.format(k, results_name)
            path = os.path.join(save_results, filename)

            print("======================= Saving results to {}".format(path))

            with open(path, 'w') as f:
                f.write(results)


def create_shap_dataframe(feature_names, shap_vals_neg, shap_vals_pos, test_split):
    # Create a pandas dataframe for ease
    # In theory these should have the same length, but it's good to just check
    num_features = max([len(l) for l in shap_vals_neg] + [len(l) for l in shap_vals_pos])
    print('num features:', num_features)
    if feature_names is None:
        columns = [str(i) for i in range(num_features)]
    else:
        columns = feature_names

    df = pd.DataFrame(shap_vals_neg + shap_vals_pos, columns=columns)
    labels = pd.DataFrame.from_records([[0] for _ in range(len(shap_vals_neg))] +
                                       [[1] for _ in range(len(shap_vals_pos))])
    df = df.fillna(0)

    # Split the data up, normalise it
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split)
    split = sss.split(df, labels)

    for train_indices, test_indices in split:
        data_train, labels_train = df.iloc[train_indices], labels.iloc[train_indices]
        data_test, labels_test = df.iloc[test_indices], labels.iloc[test_indices]

    data_train = preprocessing.scale(data_train)
    data_test = preprocessing.scale(data_test)

    return data_test, data_train, labels_test, labels_train


def train_lr(shap_vals_neg, shap_vals_pos, feature_names=None, test_split=0.3, save=None, save_results=None,
             results_name='', cv=False, n_jobs=1, verbose=0, scale=True):

    # Split the SHAP values into test and train sets
    data_test, data_train, labels_test, labels_train = create_shap_dataframe(feature_names, shap_vals_neg,
                                                                             shap_vals_pos, test_split)

    if scale:
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

    if cv:
        lr = LogisticRegressionCV(max_iter=1000, n_jobs=n_jobs, verbose=verbose)
    else:
        lr = LogisticRegression(verbose=verbose, max_iter=10000, solver='saga')
    lr.fit(data_train, labels_train)

    if save is not None:
        filename = 'lr-{}.pkl'.format(results_name)
        path = os.path.join(save, filename)

        print("======================= Saving LR to {}".format(path))

        with open(path, 'wb') as f:
            pickle.dump(lr, f)

    print("======================= Evaluating LR")

    preds = lr.predict(data_test)

    conf_matrix = confusion_matrix(labels_test, preds).tolist()
    class_report = classification_report(labels_test, preds)

    # Include the parameters if we used CV
    if cv:
        results = "{} \n\n {} \n\n {}".format(conf_matrix, class_report, lr.get_params())
    else:
        results = "{} \n\n {}".format(conf_matrix, class_report)

    print(results)

    if save_results is not None:
        filename = 'results-{}.txt'.format(results_name)
        path = os.path.join(save_results, filename)

        print("======================= Saving results to {}".format(path))

        with open(path, 'w') as f:
            f.write(results)


def train_lda(shap_vals_neg, shap_vals_pos, feature_names=None, test_split=0.3, save=None, save_results=None,
             results_name='', n_jobs=1, verbose=0, scale=True):

    # Split the SHAP values into test and train sets
    data_test, data_train, labels_test, labels_train = create_shap_dataframe(feature_names, shap_vals_neg,
                                                                             shap_vals_pos, test_split)

    if scale:
        scaler = preprocessing.StandardScaler().fit(data_train)
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

    lda = LinearDiscriminantAnalysis()
    lda.fit(data_train, labels_train)

    if save is not None:
        filename = 'lr-{}.pkl'.format(results_name)
        path = os.path.join(save, filename)

        print("======================= Saving LR to {}".format(path))

        with open(path, 'wb') as f:
            pickle.dump(lr, f)

    print("======================= Evaluating LR")

    preds = lda.predict(data_test)

    conf_matrix = confusion_matrix(labels_test, preds).tolist()
    class_report = classification_report(labels_test, preds)

    results = "{} \n\n {}".format(conf_matrix, class_report)
    print(results)

    if save_results is not None:
        filename = 'results-{}.txt'.format(results_name)
        path = os.path.join(save_results, filename)

        print("======================= Saving results to {}".format(path))

        with open(path, 'w') as f:
            f.write(results)


class MSSIM(nn.Module):
    def __init__(self,
                 in_channels=3,
                 window_size=11,
                 size_average=True) -> None:
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        [1] https://github.com/jorge-pessoa/pytorch-msssim/blob/dev/pytorch_msssim/__init__.py
            (MIT License)
        :param in_channels: (Int)
        :param window_size: (Int)
        :param size_average: (Bool)
        """
        super(MSSIM, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.size_average = size_average

    def gaussian_window(self, window_size, sigma):
        kernel = torch.tensor([exp((x - window_size // 2)**2/(2 * sigma ** 2))
                               for x in range(window_size)])
        return kernel/kernel.sum()

    def create_window(self, window_size, in_channels):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(in_channels, 1, window_size, window_size).contiguous()
        return window

    def ssim(self,
             img1,
             img2,
             window_size,
             in_channel,
             size_average):

        device = img1.device
        window = self.create_window(window_size, in_channel).to(device)
        mu1 = F.conv2d(img1, window, padding= window_size//2, groups=in_channel)
        mu2 = F.conv2d(img2, window, padding= window_size//2, groups=in_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding = window_size//2, groups=in_channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding = window_size//2, groups=in_channel) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding = window_size//2, groups=in_channel) - mu1_mu2

        img_range = 1.0 #img1.max() - img1.min() # Dynamic range
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        return ret, cs

    def forward(self, img1, img2):
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []

        for _ in range(levels):
            sim, cs = self.ssim(img1, img2,
                                self.window_size,
                                self.in_channels,
                                self.size_average)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        # if normalize:
        #     mssim = (mssim + 1) / 2
        #     mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights

        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1 - output


def gaussian(x, mu=0.0, sigma=1.0, a=1.0):
    return x.sub(mu).div(sigma).pow(2).div(2).neg().exp().mul(a)


def get_meshgrid(size, **kwargs):
    _, _, h, w = size

    y = torch.arange(h, **kwargs).float().add(0.5).div(h)
    x = torch.arange(w, **kwargs).float().add(0.5).div(w)

    grid_y, grid_x = torch.meshgrid(y, x)

    grid_y = grid_y[None, None, :, :].expand(*size)
    grid_x = grid_x[None, None, :, :].expand(*size)

    return grid_y, grid_x


def gaussian_heatmap(size, centers, sigma=1.0, a=1.0, device=None):
    grid_y, grid_x = get_meshgrid(size, device=device)

    c_y = centers[:, :, 0][:, :, None, None]
    c_x = centers[:, :, 1][:, :, None, None]

    return a * gaussian(grid_y, mu=c_y, sigma=sigma) * gaussian(
        grid_x, mu=c_x, sigma=sigma)


def normalise_sum(data):
    """
    inputs: Tensor of shape [b, c, h, w]
    """

    data_normalised = torch.zeros_like(data)
    for batch in range(len(data)):
        data_normalised[batch] = data[batch] / torch.sum(data[batch])

    return data_normalised
