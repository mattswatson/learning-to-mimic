from torch.utils.tensorboard import SummaryWriter
import os, sys, pickle

import json, argparse, random
from datetime import datetime
from tqdm import tqdm

import statsmodels.api as smtools
import scipy.stats as stats

import torch
from torchvision import datasets, transforms
from torchvision import models as pt_models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import numpy as np

from captum.attr import GradientShap

from XrayDataset import GazeXrayDataset
from UnetClassifier import UnetClassifier

from explanation_ensemble import Ensemble, get_models, save


def train(models, loader, epoch, criterion, optimizer, writer):
    for m in models:
        m.train()

    ensemble = Ensemble(models)

    losses = 0
    ensemble_losses = 0
    model_losses = [0 for _ in range(len(models))]
    correct = 0
    total = 0

    loss_fn = nn.NLLLoss()

    for batch_idx, (inputs, heatmaps, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        _, target_label = torch.max(targets, 2)

        # If we squeeze with a batch size of 1, we will remove that dimension
        if target_label.shape[0] != 1:
            target_label = torch.squeeze(target_label)
        else:
            target_label = target_label[0]

        for i in range(len(models)):
            optimizer[i].zero_grad()

        ce_loss = 0
        for i, m in enumerate(models):
            outputs = m(inputs)

            l = criterion(outputs, target_label)
            ce_loss += l
            model_losses[i] += l.item()

        outputs = ensemble(inputs)
        ensemble_loss = loss_fn(outputs, target_label)
        probs = torch.exp(outputs)
        _, predicted = probs.max(1)

        correct += predicted.eq(target_label).sum().item()
        total += inputs.size(0)

        loss = ce_loss

        losses += loss.item()
        ensemble_losses += ensemble_loss.item()

        loss.backward()

        for i in range(len(models)):
            optimizer[i].step()

    print_message = 'Epoch [%3d] | ' % epoch
    for i in range(len(models)):
        print_message += 'Model{i:d}: {loss:.4f}  '.format(
            i=i + 1, loss=model_losses[i] / (batch_idx + 1))
    tqdm.write(print_message)

    if writer:
        writer.add_scalar('train/classification_loss', losses / total, epoch)
        writer.add_scalar('train/ensemble_loss', ensemble_losses / total, epoch)
        writer.add_scalar('train/classification_accuracy', correct / total, epoch)


def test(models, test_loader, writer, epoch, verbose):
    for m in models:
        m.eval()

    ensemble = Ensemble(models)

    criterion = nn.NLLLoss()

    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, heatmaps, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.type(torch.LongTensor).cuda()
            _, target_label = torch.max(targets, 2)

            # If we squeeze with a batch size of 1, we will remove that dimension
            if target_label.shape[0] != 1:
                target_label = torch.squeeze(target_label)
            else:
                target_label = target_label[0]

            output = ensemble(inputs)
            loss += criterion(output, target_label).item()

            probs = torch.exp(output)
            _, preds = torch.max(probs, 1)
            correct += preds.eq(target_label).sum().item()

            if verbose:
                print('pred:', preds)
                print('targets:', targets)
                print('num correct:', correct)
                print('classification loss:', loss.item())
                print('Accuracy for epoch {}: {}'.format(epoch, 100 * correct / total))

            total += inputs.size(0)

    if writer:
        writer.add_scalar('test/classification_loss', loss / total, epoch)
        writer.add_scalar('test/ensemble_acc', 100 * correct / total, epoch)

    print_message = 'Evaluation  | Classification Loss {loss:.4f} Acc {acc:.2%}'.format(
        loss=loss / len(test_loader), acc=correct / total)
    tqdm.write(print_message)

    return loss / len(test_loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    model_args = parser.add_argument_group('model args')
    training_args = parser.add_argument_group('training args')
    data_args = parser.add_argument_group('dataset args')
    cxr_data_args = parser.add_argument_group('cxr dataset args')

    model_args.add_argument('--model-type', choices=['densenet', 'unet'], default='densenet',
                            help='Model architecture to use as ensemble sub-models')
    model_args.add_argument('--model-num', default=3, type=int, help='number of submodels within the ensemble')
    model_args.add_argument('--model-file', default=None, type=str,
                            help='Path to the file that contains model checkpoints')

    training_args.add_argument('--gpu', default='0', type=str, help='gpu id')
    training_args.add_argument('--seed', default=0, type=int, help='random seed for torch')

    training_args.add_argument('--epochs', default=200, type=int, help='number of training epochs')
    training_args.add_argument('--lr', default=0.1, type=float, help='learning rate')
    training_args.add_argument('--early-stopping', type=float, default=None,
                               help='Early stopping if accuracy decreases')

    data_args.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    data_args.add_argument('--batch-size', type=int, default=128, help='batch size of the train loader')
    data_args.add_argument('--dataset', choices=['mimic-cxr', 'egd'], default='mnist',
                           help='Dataset to use')
    data_args.add_argument('--shuffle', action='store_true', help='Shuffle training set')
    data_args.add_argument('--test-split', type=float, default=0.2, help='Size of test data')

    # Specific arguments for EGD dataset
    cxr_data_args.add_argument('--cxr-data-path', type=str, help='Path to JPG CXR data')
    cxr_data_args.add_argument('--gaze-data-path', type=str,  help='Path to EGD data')
    cxr_data_args.add_argument('--generated-heatmaps-path', type=str,
                               help='Path to pre-generated heatmaps. If None, generate heatmaps at runtime')
    cxr_data_args.add_argument('--label', choices=['Normal', 'CHF', 'Pneumothorax', 'all'], default='all',
                               help='Label to predict')

    parser.add_argument('--save', type=str, default=None, help='Path to save results to')
    parser.add_argument('--tb-path', type=str, default='./', help='Tensorboard results path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    parser.add_argument('--test', action='store_true', help='Test a pre-trained model')

    args = parser.parse_args()

    # set up random seed
    torch.manual_seed(args.seed)

    now = datetime.now()

    tb_name = '{}-normal_ensemble-seed_{:d}-{}'.format(args.dataset, args.seed, now.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(os.path.join(args.tb_path, 'tb', tb_name))

    # initialize models
    if args.test:
        models = get_models(args, train=True, as_ensemble=False, model_file=args.model_file)
    else:
        models = get_models(args, train=True, as_ensemble=False, model_file=None)

    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}
    criterion = nn.CrossEntropyLoss()

    if args.dataset == 'egd':
        # Normalisation for imagenet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        cxr_transforms = [transforms.Resize((224, 224)), transforms.Normalize(mean=mean, std=std)]

        dataset = GazeXrayDataset(args.gaze_data_path, args.cxr_data_path, cxr_transforms=cxr_transforms,
                                  generated_heatmaps_path=args.generated_heatmaps_path)

        indices = list(range(len(dataset)))
        split = int(np.floor(args.test_split * len(dataset)))

        np.random.shuffle(indices)

        train_indices, test_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=8, pin_memory=True)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler,
                                 num_workers=8, pin_memory=True)
    else:
        raise NotImplementedError("Dataset {} not supported!".format(args.dataset))

    params = []
    optimisers = []
    for m in models:
        params += list(m.parameters())
        optimisers.append(optim.Adam(m.parameters(), lr=args.lr))

    epoch_iterator = tqdm(list(range(1, args.epochs + 1)), total=args.epochs, desc='Epoch', leave=False, position=1)

    best_acc = 0
    verbose_test = args.verbose or args.test
    for epoch in epoch_iterator:
        train(models, train_loader, epoch, criterion, optimisers, writer)

        loss, accuracy = test(models, test_loader, writer, epoch, verbose=verbose_test)

        if args.test:
            # We only want to test once if we're not training
            break

        save(models, epoch, args)

        if args.early_stopping is not None:
            if accuracy > best_acc:
                best_acc = accuracy

            # If the accuracy has decreased too much, stop
            if best_acc - accuracy > args.early_stopping:
                print('early stopping')
                break


if __name__ == '__main__':
    main()
