import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler

import numpy as np
import segmentation_models_pytorch as smp

from captum.attr import GradientShap

# These are for our own baselines we've created
from UnetClassifier import UnetClassifier as OurUnetClassifier
import torchvision.models as pt_models
import torchvision.transforms as transforms
from train import create_model

from utils import GradCam, compute_gradCAM
from XrayDataset import GazeXrayDataset
from explanation_ensemble import get_models, Ensemble


def main():
    parser = argparse.ArgumentParser(description='Compute and save GradCAM or SHAP attributions for a given model')

    parser.add_argument('--model-path', type=str, help='Path to model checkpoint to load', required=True)
    parser.add_argument('--model-type', default='densenet', choices=['densenet', 'unet'],
                        help='Model architecture to use as ensemble sub-models')
    parser.add_argument('--model-teacher', type=str, default='timm-efficientnet-b0', help='model_teacher')
    parser.add_argument('--pretrained-name', type=str, default='noisy-student', help='model pretrained value')

    parser.add_argument('--expl-type', choices=['shap', 'gradcam'], default='shap', help='Explanation method to use')
    parser.add_argument('--expl-per-submodel', action='store_true',
                        help='Calculate separate expalanations for each submodels in the ensemble')

    parser.add_argument('--dataset', choices=['egd'], default='egd')
    parser.add_argument('--cxr-data-path', type=str,  help='Path to JPG CXR data')
    parser.add_argument('--gaze-data-path', type=str, help='Path the EGD data')
    parser.add_argument('--generated-heatmaps-path', type=str,
                        help='Path to pre-generated heatmaps. If None, generate heatmaps at runtime')
    parser.add_argument('--split', choices=['test', 'train'], help='Split of dataset to use', default='test')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size to use')

    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()
    args.verbose = False

    np.random.seed(42)
    torch.manual_seed(42)

    # Load the model
    models = get_models(args, train=True, as_ensemble=False, model_file=args.model_path)
    model = Ensemble(models)

    model = model.cuda()
    model.eval()

    if args.model_type == 'baseline-densenet':
        candidate_layers = ['classifier']
        target_layer = 'classifier'
    else: # Unet
        candidate_layers = [f'encoder.blocks.{i}' for i in range(0, 7)]
        target_layer = 'encoder.blocks.6'

    if args.expl_type == 'shap':
        if args.expl_per_submodel:
            shap = [GradientShap(m) for m in model.models]
        else:
            shap = GradientShap(model)
    else:
        # The final GradCAM output of the Ensemble is the weighted average of GradCAM for each of the sub-models
        # Using the same weighting as the prediction
        gcams = []
        for i in range(len(models)):
            gcams.append(GradCam(model=models[i], candidate_layers=candidate_layers))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    cxr_transforms = [transforms.Resize((224, 224)), transforms.Normalize(mean=mean, std=std)]
    dataset = GazeXrayDataset(args.gaze_data_path, args.cxr_data_path, cxr_transforms=cxr_transforms,
                              generated_heatmaps_path=args.generated_heatmaps_path)

    indices = list(range(len(dataset)))
    split = int(np.floor(0.2 * len(dataset)))

    np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=12, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler,
                                              num_workers=12, pin_memory=True)

    if args.split == 'test':
        dataset = test_loader
    else:
        dataset = train_loader

    all_expl = []
    for images, heatmaps, labels in tqdm(dataset):
        images = images.cuda()
        labels = labels.cuda()
        _, target_label = torch.max(labels, 2)
        target_labels = torch.squeeze(target_label)

        if args.expl_type == 'gradcam':
            prob_activation = nn.Sigmoid()

            output = model(images)

            expl_outs = []
            for i in range(len(models)):
                _, expl_out, one_hot = compute_gradCAM(output, labels, gcams[i], False, prob_activation, target_layer)
                expl_outs.append(expl_out)

            if args.expl_per_submodel:
                expl_out = expl_outs
            else:
                expl_out = torch.mean(torch.stack(expl_outs), dim=0)
        elif args.expl_type == 'shap':
            # As we're using images it should be fine to use the all zero tensor as a baseline
            baseline = torch.zeros_like(images)
            baseline = baseline.cuda()

            if args.expl_per_submodel:
                expl_outs = []
                for i in range(len(model.models)):
                    expl = shap[i].attribute(images, baselines=baseline, target=target_labels)
                    expl_outs.append(expl)

                expl_out = expl_outs
            else:
                expl_out = shap.attribute(images, baselines=baseline, target=target_labels)
        else:
            raise NotImplementedError('Explanation technique {} not supported!'.format(args.expl_type))

        for i in range(len(expl_out)):
            all_expl.append(expl_out[i])

    with open(os.path.join(args.output_dir, '{}.pt'.format(args.expl_type)), 'wb') as f:
        torch.save(torch.stack(all_expl), f)

    print('----------- {} saved to {}'.format(args.expl_type.upper(), os.path.join(args.output_dir,
                                                                                   '{}.pt'.format(args.expl_type))))


if __name__ == '__main__':
    main()
