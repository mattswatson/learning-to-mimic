import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

from captum.attr import GradientShap

from main_Unet import load_data
from models.classifier import UnetClassifier
from utils.gradcam_utils import GradCam, compute_gradCAM

from train import create_model

def main():
    parser = argparse.ArgumentParser(description='Compute and save GradCAM or SHAP attributions for a given model')

    parser.add_argument('--model-path', type=str, help='Path to model checkpoint to load', required=True)
    parser.add_argument('--model-type', default='unet', choices=['baseline-kara', 'unet-static', 'baseline-densenet',
                                                                 'baseline-unet'],
                        help='baseline-kara: baseline from paper; unet-satic: use static heatmaps, from paper;'
                             'baseline-densenet: densenet, no heatmaps; baseline-unet: unet, no heatmaps')
    parser.add_argument('--model-teacher', type=str, default='timm-efficientnet-b0', help='model_teacher')
    parser.add_argument('--pretrained-name', type=str, default='noisy-student', help='model pretrained value')

    parser.add_argument('--expl-type', choices=['gradcam', 'shap'], default='gradcam', help='Explanation method to use')

    parser.add_argument('--data-path', type=str, help='Data path')
    parser.add_argument('--image-path', type=str, help='image_path')
    parser.add_argument('--heatmaps-path', type=str, help='Heatmaps directory')
    parser.add_argument('--split', choices=['test', 'train', 'val'], help='Split of dataset to use', default='test')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size to use')

    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    # Load the model
    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=None,  # dropout ratio, default is None
        activation=None,  # activation function, default is None
        classes=3,  # define number of output labels
    )

    if args.model_type == 'baseline-kara':
        model = UnetClassifier(encoder_name=args.model_teacher, classes=1, encoder_weights=args.pretrained_name,
                               aux_params=aux_params).to('cuda:0')
    elif args.model_type == 'unet-static':
        model = smp.Unet(args.model_teacher, classes=1, encoder_weights=args.pretrained_name,
                         aux_params=aux_params).to('cuda:0')
    elif args.model_type in ['baseline-densenet', 'baseline-unet']:
        model_type = args.model_type.replace('baseline-', '')
        model = create_model(model_type, 3, label='all')
    else:
        raise NotImplementedError("Model architecture {} is not supported!".format(args.model_type))

    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    if args.expl_type == 'shap':
        if args.model_type != 'baseline-densenet':
            # As we're using Unet, our output is the generated mask and thee classification
            # We care only about the classification here
            def forward_pass(x):
                masks_pred, y_cl = model(x)

                return y_cl

            shap = GradientShap(forward_pass)
        else:
            shap = GradientShap(model)

    candidate_layers = [f'encoder.blocks.{i}' for i in range(0, 7)]
    gcam = GradCam(model=model, candidate_layers=candidate_layers)

    # Load the dataset
    train_dl, valid_dl, test_dl = load_data(args.model_type, args.data_path, args.image_path, args.heatmaps_path,
                                            224, ['Normal', 'CHF', 'pneumonia'], args.batch_size, 16, 42, None)

    if args.split == 'test':
        dataset = test_dl
    elif args.plit == 'val':
        dataset = valid_dl
    else:
        dataset = train_dl

    all_expl = []
    for images, labels, idx, x_hm, y_hm in tqdm(dataset):
        images = images.cuda()
        labels = labels.cuda()
        y_hm = y_hm.cuda()

        if args.expl_type == 'gradcam':
            prob_activation = nn.Sigmoid()

            masks_pred, y_cl = model(images)
            _, expl_out, one_hot = compute_gradCAM(y_cl, labels, gcam, False, prob_activation, 'encoder.blocks.6')
        elif args.expl_type == 'shap':
            # As we're using images it should be fine to use the all zero tensor as a baseline
            baseline = torch.zeros_like(images)
            baseline = baseline.cuda()

            # labels is a one-hot encoded label vector, we want just the index of the target class
            _, target_labels = torch.max(labels, axis=-1)

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
