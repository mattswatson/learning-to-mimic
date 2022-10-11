import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler

import numpy as np
import segmentation_models_pytorch as smp

from captum.attr import GradientShap, GuidedGradCam

# These are for our own baselines we've created
from UnetClassifier import UnetClassifier as OurUnetClassifier
import torchvision.models as pt_models
import torchvision.transforms as transforms
from train import create_model

from utils import GradCam, compute_gradCAM
from XrayDataset import GazeXrayDataset


def main():
    parser = argparse.ArgumentParser(description='Compute and save GradCAM or SHAP attributions for a given model')

    parser.add_argument('--model-path', type=str, help='Path to model checkpoint to load', required=True)
    parser.add_argument('--model-type', default='unet', choices=['baseline-kara', 'unet-static', 'baseline-densenet',
                                                                 'baseline-unet'],
                        help='baseline-kara: baseline from paper; unet-satic: use static heatmaps, from paper;'
                             'baseline-densenet: densenet, no heatmaps; baseline-unet: unet, no heatmaps')
    parser.add_argument('--model-teacher', type=str, default='timm-efficientnet-b0', help='model_teacher')
    parser.add_argument('--pretrained-name', type=str, default='noisy-student', help='model pretrained value')

    parser.add_argument('--expl-type', choices=['gradcam', 'shap', 'c-gradcam'], default='gradcam',
                        help='Explanation method to use')

    parser.add_argument('--cxr-data-path', type=str, help='Path to JPG CXR data')
    parser.add_argument('--gaze-data-path', type=str, help='Path the EGD data')
    parser.add_argument('--generated-heatmaps-path', type=str,
                        help='Path to pre-generated heatmaps. If None, generate heatmaps at runtime')
    parser.add_argument('--split', choices=['test', 'train'], help='Split of dataset to use', default='test')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size to use')

    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    # Load the model
    if args.model_type == 'baseline-kara':
        raise NotImplementedError("Model architecture {} is not supported! "
                                  "Use egd_code/Experiments/compute_explanations instead".format(args.model_type))
    elif args.model_type == 'unet-static':
        raise NotImplementedError("Model architecture {} is not supported! "
                                  "Use egd_code/Experiments/compute_explanations instead".format(args.model_type))
    elif args.model_type in ['baseline-densenet', 'baseline-unet']:
        model_type = args.model_type.replace('baseline-', '')
        model = create_model(model_type, 3, label='all')
    else:
        raise NotImplementedError("Model architecture {} is not supported!".format(args.model_type))

    np.random.seed(42)
    torch.manual_seed(42)

    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    if args.expl_type == 'shap':
        shap = GradientShap(model)

    # Different layers will be target by GradCAM depending on the model used
    if args.model_type == 'baseline-densenet':
        candidate_layers = ['classifier']
        target_layer = 'classifier'

        if args.expl_type == 'c-gradcam':
            guided_gcam = GuidedGradCam(model, model.classifier)
    else: # Unet
        candidate_layers = [f'encoder.blocks.{i}' for i in range(0, 7)]
        target_layer = 'encoder.blocks.6'

        if args.expl_type == 'c-gradcam':
            guided_gcam = GuidedGradCam(model, model.encoder.blocks[6])

    gcam = GradCam(model=model, candidate_layers=candidate_layers)

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
            _, expl_out, one_hot = compute_gradCAM(output, labels, gcam, False, prob_activation, target_layer)
        elif args.expl_type == 'shap':
            # As we're using images it should be fine to use the all zero tensor as a baseline
            baseline = torch.zeros_like(images)
            baseline = baseline.cuda()

            expl_out = shap.attribute(images, baselines=baseline, target=target_labels)
        elif args.expl_type == 'c-gradcam':
            expl_out = guided_gcam.attribute(images, target=target_labels)
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
