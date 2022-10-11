import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision.models as pt_models

import os
import argparse
import random
import visdom
import tqdm

import pandas as pd
import numpy as np

from XrayDataset import GazeXrayDataset
from utils import AverageMeter, VisdomLinePlotter
from UnetClassifier import UnetClassifier


def train(model, loader, optimiser, loss_fn, device):
    losses = AverageMeter()
    model.train()

    with tqdm.tqdm(total=len(loader), position=1) as progress:
        for inputs, _, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, target_label = torch.max(labels, 2)
            target_label = torch.squeeze(target_label)

            optimiser.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, target_label)

            losses.update(loss.data, len(labels))

            loss.backward()
            optimiser.step()

            progress.update(1)

    return losses.avg.cpu()


def test(model, loader, loss_fn, device):
    losses = AverageMeter()
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm.tqdm(total=len(loader), position=1) as progress:
            for inputs, _, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                _, target_label = torch.max(labels, 2)
                target_label = torch.squeeze(target_label)

                outputs = model(inputs)

                loss = loss_fn(outputs, target_label)

                probs = F.softmax(outputs, 1)
                _, preds = torch.max(probs, 1)

                losses.update(loss.data, len(labels))

                correct += torch.sum(preds == target_label).item()
                total += len(labels)

                progress.update(1)

    acc = (correct / total) * 100

    return losses.avg.cpu(), acc


def create_model(model_type, num_classes, label='all'):
    """
    :param model_type: string for model type to load, 'densenet' or 'unet'
    :param num_classes: int, number of possible classification labels
    :param label: which set of labels do we want to classify? ['Normal', 'CHF', 'Pneumothorax', 'all']

    :return model to finetune
    """
    if model_type == 'densenet':
        # Load a pre-trained densenet model
        model = pt_models.densenet121(pretrained=True)

        # We need to change the last layer of the network to output the correct number of classes
        binary_classification = not (label == 'all')
        num_in_last_layer = model.classifier.in_features
        if binary_classification:
            model.classifier = nn.Linear(num_in_last_layer, 2)
        else:
            model.classifier = nn.Linear(num_in_last_layer, 3)
    elif model_type == 'unet':
        model = UnetClassifier(
            encoder_name='timm-efficientnet-b0',  # The default used by the EGD paper, Pytorch Image Models EN
            encoder_depth=5,
            encoder_weights='imagenet',  # Use model pretrained on imagenet
            in_channels=3,
            num_classes=num_classes
        )
    else:
        raise NotImplementedError("Model architecture {} is not supported!".format(model_type))

    return model


def main():
    """
    Finetune a Densenet 121 model on the CXR EGD dataset. For now, we're only using the images
    """
    parser = argparse.ArgumentParser(description='Finetune pre-trained (on imagenet) architectures to classify CXR '
                                                 'images using gaze data')

    dataset_args = parser.add_argument_group("Dataset arguments")
    model_args = parser.add_argument_group("Model/training arguments")
    plot_args = parser.add_argument_group("Plotting arguments")

    dataset_args.add_argument('--cxr-data-path', type=str, help='Path to JPG CXR data')
    dataset_args.add_argument('--gaze-data-path', type=str, default=None, help='Path the EGD data')
    dataset_args.add_argument('--generated-heatmaps-path', type=str, default=None,
                              help='Path to pre-generated heatmaps. If None, generate heatmaps at runtime')
    dataset_args.add_argument('--label', choices=['Normal', 'CHF', 'Pneumothorax', 'all'], default='all',
                              help='Label to predict')

    model_args.add_argument('--model-type', choices=['densenet', 'unet'], default='densenet',
                            help='Model architecture to use for classification')
    model_args.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    model_args.add_argument('--batch-size', type=int, default=32, help='Batch size during training')
    model_args.add_argument('--lr', type=float, default=0.001, help='Learning rate during training')
    model_args.add_argument('--test-split', type=float, default=0.2, help='Proportion of dataset to use for testing')
    model_args.add_argument('--save-dir', type=str, default=None, help='Directory to save model checkpoints to')

    plot_args.add_argument('--visdom-server', type=str, default='localhost', help='URL to Visdom server')
    plot_args.add_argument('--visdom-port', type=int, default=8097, help='Visdom server port')
    plot_args.add_argument('--plot', type=str, default=None, help='Name of Visdom plot')

    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.plot:
        plotter = VisdomLinePlotter(args.plot, server=args.visdom_server, port=args.visdom_port)

    if args.label == 'all':
        num_classes = 3
    else:
        num_classes = 1

    model = create_model(args.model_type, num_classes, label=args.label)

    model = model.to(device)

    # Load the data
    # Normalisation for imagenet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    cxr_transforms = [transforms.Resize(224), transforms.Normalize(mean=mean, std=std)]

    dataset = GazeXrayDataset(args.gaze_data_path, args.cxr_data_path, cxr_transforms=cxr_transforms,
                              generated_heatmaps_path=args.generated_heatmaps_path)

    indices = list(range(len(dataset)))
    split = int(np.floor(args.test_split * len(dataset)))

    np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=12,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler,
                                              num_workers=12,
                                              pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    with tqdm.tqdm(total=args.epochs, position=0) as progress:
        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimiser, criterion, device)
            test_loss, acc = test(model, test_loader, criterion, device)

            print('Epoch {}: [Train loss: {:.4f}] [Test loss: {:.4f} | Test acc: {:.4f}]'.format(epoch, train_loss,
                                                                                                 test_loss, acc))

            if args.plot:
                plotter.plot('loss', 'train', 'Classification loss', epoch, train_loss)
                plotter.plot('loss', 'test', 'Classification Loss', epoch, test_loss)
                plotter.plot('acc', 'test', 'Classification Accuracy', epoch, acc)

            if args.save_dir:
                save_dir = os.path.join(args.save_dir, '{}-epochs{}-bs{}-lr{}-labels{}'.format(args.model_type,
                                                                                               args.epochs,
                                                                                               args.batch_size,
                                                                                               args.lr,
                                                                                               args.label))

                os.makedirs(save_dir, exist_ok=True)
                model_save_path = os.path.join(save_dir, 'model-checkpoint{}.pth'.format(epoch))
                optim_save_path = os.path.join(save_dir, 'optim-checkpoint{}.pth'.format(epoch))
                print('---------- Saving model to {}'.format(model_save_path))

                torch.save(model.state_dict(), model_save_path)
                torch.save(optimiser, optim_save_path)

            progress.update(1)


if __name__ == '__main__':
    main()