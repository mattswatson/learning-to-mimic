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

from efficientnet_pytorch import EfficientNet


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            for model in self.models:
                outputs += F.log_softmax(model(x), dim=-1)
            output = outputs / len(self.models)
            #output = torch.clamp(output, min=1e-40)

            #print(output)
            return output
        else:
            return F.log_softmax(self.models[0](x), dim=-1)


class SHAPDataset(Dataset):
    def __init__(self, shap):
        self.samples = torch.zeros(shap.shape[0] * shap.shape[1], shap.shape[2], dtype=torch.float32)
        self.labels = torch.zeros(shap.shape[0] * shap.shape[1], dtype=torch.float32)

        sample_start = 0
        for sample in shap:
            self.samples[sample_start:sample_start+shap.shape[1]] = sample
            self.labels[sample_start:sample_start+shap.shape[1]] = torch.tensor(range(0, shap.shape[1]))
            sample_start += shap.shape[1]

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        return self.samples[item], self.labels[item]


# Make this an MLP to begin with
class Discriminator(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_classes=3):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def get_models(args, train=True, as_ensemble=False, model_file=None):
    models = []

    if model_file:
        state_dict = torch.load(model_file)
        if train:
            print('Loading pre-trained models...')

    iter_m = state_dict.keys() if model_file else range(args.model_num)

    if args.dataset == 'egd':
        num_classes = 3
        input_size = 224 * 244
    else:
        raise NotImplementedError("Dataset {} not supported!".format(args.dataset))

    for i in iter_m:
        if args.dataset == 'mimic-cxr':
            raise NotImplementedError("Dataset {} not supported!".format(args.dataset))
        elif args.dataset == 'egd':
            if args.model_type == 'densenet':
                model = pt_models.densenet121(pretrained=True)

                num_in_last_layer = model.classifier.in_features
                model.classifier = nn.Linear(num_in_last_layer, 3)
            elif args.model_type == 'efficientnet':
                model = EfficientNet.from_pretrained('efficientnet-b0')

                # Much like for densenet, we want the last layer to only output 3 class probabilties
                # We also want logsoftmax so we can use NLL loss
                num_in_last_layer = model._fc.in_features
                model._fc = nn.Linear(num_in_last_layer, 3)
            else:
                model = UnetClassifier(
                    encoder_name='timm-efficientnet-b0',  # The default used by the EGD paper, Pytorch Image Models EN
                    encoder_depth=5,
                    encoder_weights='imagenet',  # Use model pretrained on imagenet
                    in_channels=3,
                    num_classes=3
                )
        else:
            raise NotImplementedError('Dataset {} not supported!'.format(args.dataset))

        if args.verbose:
            print(model)

        if model_file:
            model.load_state_dict(state_dict[i])

        if train:
            model = model.train()
        else:
            model = model.eval()

        model = model.cuda()
        models.append(model)

    if as_ensemble:
        assert not train, 'Must be in eval mode when getting models to form an ensemble'
        ensemble = Ensemble(models)
        ensemble.eval()
        return ensemble
    else:
        return models


def get_batch_shap(models, data, targets, args, baseline_method='mean', flatten=False, threshold=None, save=None,
                   epoch=0, normalise=False):
    if baseline_method == 'mean':
        baseline = torch.zeros_like(data)
        baseline = baseline.fill_(0.1307)
    else:
        raise NotImplementedError('Baseline method {} not yet implemented'.format(baseline_method))

    all_models_shap = []

    for i, m in enumerate(models):
        expl = GradientShap(m)

        # Shape: batch_size * data_size
        shap_vals = expl.attribute(data, baselines=baseline, target=targets)
        shap_vals = torch.abs(shap_vals)

        if normalise:
            for i in range(shap_vals.shape[0]):
                minimum = torch.min(shap_vals[i])
                r = torch.max(shap_vals[i]) - minimum

                if r > 0:
                    shap_vals[i] = (shap_vals[i] - minimum) / r
                else:
                    shap_vals[i] = torch.zeros_like(shap_vals[i])

        shap_vals_flat = torch.stack([torch.abs(s.flatten()) for s in shap_vals])

        if torch.isnan(shap_vals_flat).any():
            print(shap_vals_flat)
            raise Exception('shap_vals contains nan')

        if threshold is not None:
            flat = shap_vals.cpu().numpy().flatten()

            ecdf = smtools.distributions.ECDF(flat)
            idx = [i for i in range(len(ecdf.y)) if ecdf.y[i] >= threshold][0]
            x_val = ecdf.x[idx]

            if args.threshold_larger:
                shap_vals_flat[shap_vals_flat >= x_val] = 0
            else:
                shap_vals_flat[shap_vals_flat < x_val] = 0

        if flatten:
            shap_vals = shap_vals_flat

        if save is not None:
            os.makedirs(os.path.join(save, 'model_{}'.format(i)), exist_ok=True)
            path = os.path.join(save, 'model_{}'.format(i), 'epoch{}.pkl'.format(epoch))

            with open(path, 'wb') as f:
                pickle.dump(shap_vals, f)

        all_models_shap.append(shap_vals.cpu())

    # Shape: num_models * batch_size * data_size
    all_models_shap_tensor = torch.stack(all_models_shap)

    return all_models_shap_tensor


def train_discriminator(shap, model, loss_fn, optimiser, epoch, batch_num, writer):
    shap_dataset = SHAPDataset(shap)

    indices = list(range(len(shap_dataset)))
    random.shuffle(indices)
    train_indices = indices[:int(0.8 * len(shap_dataset))]
    test_indices = indices[int(0.8 * len(shap_dataset)):]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(shap_dataset, batch_size=4, shuffle=False,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(shap_dataset, batch_size=4, shuffle=False,
                                             sampler=test_sampler)

    model = model.cuda()
    model.train()
    train_losses = 0
    for data, labels in train_loader:
        data = data.float().cuda()
        labels = labels.long().cuda()

        optimiser.zero_grad()
        output = model(data)
        train_loss = loss_fn(output, labels)

        train_losses += train_loss

        train_loss.backward()
        optimiser.step()

    if writer:
        writer.add_scalar('train/discrim_loss_epoch{}'.format(epoch), train_losses / len(train_loader), batch_num)

    return train_losses / len(train_loader)


def train_discriminator_batch(shap, model, loss_fn, optimiser):
    shap_dataset = SHAPDataset(shap)

    model = model.cuda()
    model.train()

    data = torch.tensor(shap_dataset.samples).float().cuda()
    targets = torch.tensor(shap_dataset.labels).long().cuda()

    if optimiser is None:
        output = model(data)
        train_loss = loss_fn(output, targets)
    else:
        optimiser.zero_grad()
        output = model(data)
        train_loss = loss_fn(output, targets)
        train_loss.backward()
        optimiser.step()

    return train_loss


def get_average_shap(models, loader, save_path, epoch, args):
    data_flat_shape = next(iter(loader))[0][0].flatten().shape

    shap = torch.zeros((len(loader) * args.batch_size, len(models), data_flat_shape[0]))
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # batch_shap: batch x num_models x flat input
        batch_shap = get_batch_shap(models, inputs, targets, args, flatten=True, threshold=args.threshold)
        batch_shap = batch_shap.permute(1, 0, 2)
        # batch_shap: num_models x batch x flat input

        start = batch_idx * batch_shap.shape[0]
        end = (batch_idx + 1) * batch_shap.shape[0]
        shap[start:end, :, :] = batch_shap

    avg_shap = torch.mean(torch.abs(shap), dim=0)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        path = os.path.join(save_path, 'epoch{}_shap.pkl'.format(epoch))
        with open(path, 'wb') as f:
            pickle.dump(avg_shap, f)


def train(models, loader, epoch, criterion, optimizer, scheduler, writer, args, seeds=None, discrim=None,
          discrim_optim=None):
    for m in models:
        m.train()

    ensemble = Ensemble(models)

    losses = 0
    ce_losses = 0
    model_losses = [0 for _ in range(len(models))]
    discrim_losses = 0
    correct = 0
    total = 0

    # This will be the shape of the SHAP values
    data_flat_shape = next(iter(loader))[0][0].flatten().shape

    loss_fn = nn.NLLLoss()

    if discrim is None:
        discrim = Discriminator(input_size=data_flat_shape[0], num_classes=len(models))
        discrim_optim = optim.Adam(discrim.parameters(), lr=0.001)

    submodel_correct = [0 for _ in range(len(models))]
    all_targets = []
    for batch_idx, (inputs, heatmaps, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        all_targets += list(targets.cpu().numpy().flatten())
        _, target_label = torch.max(targets, 2)

        # If we squeeze with a batch size of 1, we will remove that dimension
        if target_label.shape[0] != 1:
            target_label = torch.squeeze(target_label)
        else:
            target_label = target_label[0]

        ensemble_shap = get_batch_shap(models, inputs, targets, args, flatten=True, threshold=args.threshold,
                                       normalise=True)

        # ensemble_shap: batch_size x num_models x flat input
        ensemble_shap = ensemble_shap.permute(1, 0, 2)

        for i in range(len(models)):
            optimizer[i].zero_grad()
        discrim_optim.zero_grad()

        ce_loss = 0
        all_outputs = None
        for i, m in enumerate(models):
            if args.use_different_seeds:
                torch.manual_seed(seeds[i])

            outputs = m(inputs)

            l = criterion(outputs, target_label)
            ce_loss += l
            model_losses[i] += l.item()

            with torch.no_grad():
                probs = F.softmax(outputs, 1)
                _, preds = torch.max(probs, 1)

                #print(pred)
                submodel_correct[i] += preds.eq(target_label.view_as(preds)).sum().item()

        output = ensemble(inputs)
        ensemble_loss = loss_fn(output, target_label)
        probs = torch.exp(output)
        _, predicted = probs.max(1)

        correct += predicted.eq(target_label).sum().item()
        total += inputs.size(0)

        discrim_labels = torch.cat([i * torch.ones(inputs.shape[0]) for i in range(len(models))]).long().cuda()

        if epoch % args.step == 1:
            discrim_out = discrim(ensemble_shap)
        else:
            discrim_out = discrim(ensemble_shap.detach())

        discrim_loss = loss_fn(discrim_out, discrim_labels)

        if epoch % args.step == 1:
            loss = ce_loss - args.beta * discrim_loss
        else:
            loss = ce_loss

        discrim_losses += discrim_loss.item()

        losses += loss.item()
        ce_losses += ensemble_loss.item()

        loss.backward()

        for i in range(len(models)):
            optimizer[i].step()

        if epoch % args.step == 0:
            discrim_loss.backward()
            discrim_optim.step()

    for i in range(len(models)):
        print('model {} has an accuracy of {}'.format(i, submodel_correct[i]/ total))

    print('ensemble model has an accuracy of {}'.format(correct / total))
    #for i in range(len(models)):
    #    print('model {} has accuracy of {}'.format(i, submodel_correct[i] / total))

    if args.shap_interval != 0 and epoch % args.shap_interval == 0:
        get_average_shap(models, loader, args.save, epoch, args)

    print_message = 'Epoch [%3d] | ' % epoch
    for i in range(len(models)):
        print_message += 'Model{i:d}: {loss:.4f}  '.format(
            i=i + 1, loss=model_losses[i] / (batch_idx + 1))
    tqdm.write(print_message)

    if writer:
        writer.add_scalar('train/discrim_loss', discrim_losses / total, epoch)
        writer.add_scalar('train/ensemble_loss', losses / total, epoch)
        writer.add_scalar('train/classification_loss', ce_losses / total, epoch)
        writer.add_scalar('train/classification_accuracy', correct / total, epoch)


def test(models, loader, criterion, writer, epoch, verbose=False):
    for m in models:
        m.eval()

    ensemble = Ensemble(models)
    criterion = nn.NLLLoss()

    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, heatmaps, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.type(torch.LongTensor).cuda()
            _, target_label = torch.max(targets, 2)

            # If we squeeze with a batch size of 1, we will remove that dimension
            if target_label.shape[0] != 1:
                target_label = torch.squeeze(target_label)
            else:
                target_label = target_label[0]

            output = ensemble(inputs)
            print('output shape:', output.shape)
            loss += criterion(output, target_label).item()

            #probs = F.softmax(output, 1)
            probs = torch.exp(output)
            _, preds = torch.max(probs, 1)
            correct += preds.eq(target_label).sum().item()

            total += inputs.size(0)

            if verbose:
                print('probs:', probs)
                print('pred:', preds)
                print('targets:', target_label)
                print('num correct:', correct)
                print('classification loss:', loss)
                print('Accuracy for epoch {}: {}'.format(epoch, 100 * correct / total))

    if writer:
        writer.add_scalar('test/classification_loss', loss / total, epoch)
        writer.add_scalar('test/ensemble_acc', 100 * correct / total, epoch)
        writer.add_scalar('test/classification_acc', 100 * correct / total, epoch)

    print_message = 'Epoch {} [Classification loss {}] [Ensemble Accuracy {}] [Classification accuracy {}]'.format(
        epoch, loss / total, 100 * (correct / total), 100 * (correct / total)
    )
    tqdm.write(print_message)

    if writer:
        writer.add_scalar('test/classification_loss', loss / total, epoch)
        writer.add_scalar('test/ensemble_acc', 100 * correct / total, epoch)

    print_message = 'Evaluation  | Classification Loss {loss:.4f} Acc {acc:.2%}'.format(
        loss=loss / len(loader), acc=correct / total)
    tqdm.write(print_message)

    return loss / len(loader), correct / total


def save(models, epoch, args):
    if args.save is not None:
        state_dict = {}
        for i, m in enumerate(models):
            state_dict['model_%d' % i] = m.state_dict()
        torch.save(state_dict, os.path.join(args.save, '{}-epoch_{}.pth'.format(args.dataset, epoch)))


def main():
    parser = argparse.ArgumentParser()
    model_args = parser.add_argument_group('model args')
    training_args = parser.add_argument_group('training args')
    data_args = parser.add_argument_group('dataset args')
    cxr_data_args = parser.add_argument_group('cxr dataset args')

    model_args.add_argument('--model-type', choices=['densenet', 'unet', 'efficientnet'], default='densenet',
                            help='Model architecture to use as ensemble sub-models')
    model_args.add_argument('--model-num', default=3, type=int, help='number of submodels within the ensemble')
    model_args.add_argument('--model-file', default=None, type=str,
                       help='Path to the file that contains model checkpoints')
    model_args.add_argument('--threshold', type=float, default=None, help='Threshold to use if using')
    model_args.add_argument('--threshold-larger', action='store_true',
                       help='Remove pixels greater than threshold rather than smaller (i.e. keep unimportant pixels)')
    model_args.add_argument('--shap-interval', type=int, default=0, help='Save SHAP every x epochs')
    model_args.add_argument('--use-different-seeds', action='store_true',
                       help='Use different seeds when training sub-models')


    training_args.add_argument('--gpu', default='0', type=str, help='gpu id')
    training_args.add_argument('--seed', default=0, type=int, help='random seed for torch')
    training_args.add_argument('--step', type=int, default=2, help='Train discriminator every x epochs')

    training_args.add_argument('--epochs', default=200, type=int, help='number of training epochs')
    training_args.add_argument('--lr', default=0.1, type=float, help='learning rate')
    training_args.add_argument('--sch-intervals', nargs='*', default=[100, 150], type=int,
                               help='learning scheduler milestones')
    training_args.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate decay ratio')
    training_args.add_argument('--beta', type=float, default=1.0, help='Coefficient for discrim loss')
    training_args.add_argument('--lr-discrim', type=float, default=0.0001, help='LR for discriminator')
    training_args.add_argument('--keep-discrim', action='store_true',
                               help='Keep same discriminator throughout all epochs')
    training_args.add_argument('--same-opt', action='store_true',
                               help='Use same optimiser for discriminator and ensemble')
    training_args.add_argument('--joint', action='store_true', help='Train the classifier and discriminator jointly')
    training_args.add_argument('--update-discrim-rate', type=int, default=1, help='Update discriminator every x epochs')
    training_args.add_argument('--early-stopping', type=float, default=None,
                               help='Early stopping if accuracy decreases')

    data_args.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    data_args.add_argument('--batch-size', type=int, default=128, help='batch size of the train loader')
    data_args.add_argument('--dataset', choices=['mimic-cxr', 'egd'], default='mnist',
                           help='Dataset to use')
    data_args.add_argument('--shuffle', action='store_true', help='Shuffle training set')
    data_args.add_argument('--test-split', type=float, default=0.2, help='Size of test data')

    # Specific arguments for EGD dataset
    cxr_data_args.add_argument('--cxr-data-path', type=str,  help='Path to JPG CXR data')
    cxr_data_args.add_argument('--gaze-data-path', type=str, help='Path to EGD data')
    cxr_data_args.add_argument('--generated-heatmaps-path', type=str,
                               help='Path to pre-generated heatmaps. If None, generate heatmaps at runtime')
    cxr_data_args.add_argument('--label', choices=['Normal', 'CHF', 'Pneumothorax', 'all'], default='all',
                               help='Label to predict')

    parser.add_argument('--save', type=str, default=None, help='Path to save results to')
    parser.add_argument('--tb-path', type=str, default='./', help='Tensorboard results path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    parser.add_argument('--test', action='store_true', help='Test a pre-trained model')

    args = parser.parse_args()

    if args.label == 'all':
        num_classes = 3
    else:
        num_classes = 1

    if args.same_opt and not args.keep_discrim:
        raise ValueError('Cannot use same optimiser for discriminator if not using same discriminator throughout '
                         'training (i.e. --keep-discrim not passed)')

    # set up random seed
    torch.manual_seed(args.seed)

    now = datetime.now()

    tb_name = '{}-expl_ensemble-seed_{:d}-{}'.format(args.dataset, args.seed, now.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(os.path.join(args.tb_path, 'tb', tb_name))

    # initialize models
    if args.test:
        models = get_models(args, train=True, as_ensemble=False, model_file=args.model_file)
    else:
        models = get_models(args, train=True, as_ensemble=False, model_file=None)

    seeds = None
    if args.use_different_seeds:
        seeds = [random.randint(0, 500) for _ in range(len(models))]

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

    # Shape of the explanations passed to the discriminator
    data_flat_shape = next(iter(train_loader))[0][0].flatten().shape[0]

    discriminator = Discriminator(input_size=data_flat_shape, num_classes=len(models)).to(device)
    discrim_optim = optim.Adam(discriminator.parameters(), lr=args.lr_discrim)

    params = []
    optimisers = []
    for m in models:
        params += list(m.parameters())
        optimisers.append(optim.Adam(m.parameters(), lr=args.lr))

    if args.same_opt and args.keep_discrim:
        print('Using same optimiser for discriminator and ensemble')
        params += list(discriminator.parameters())

    scheduler = None

    #criterion = nn.CrossEntropyLoss()

    epoch_iterator = tqdm(list(range(1, args.epochs + 1)), total=args.epochs, desc='Epoch', leave=False, position=1)

    best_acc = 0
    verbose_test = args.verbose or args.test
    for epoch in epoch_iterator:
        train(models, train_loader, epoch, criterion, optimisers, scheduler, writer, args, seeds=seeds,
              discrim=discriminator, discrim_optim=discrim_optim)

        loss, accuracy = test(models, test_loader, criterion, writer, epoch, verbose=verbose_test)

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