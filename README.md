# Learning to MIMIC: Using Model Explanations to Guide Deep Learning Training

This is the support repository for our paper Agree to Disagree: When Deep Learning Models with Identical Architectures 
Produce Distinct Explanations, M Watson, B Hasan, N Al Moubayed (2022) to be presented at WACV 2023.

Code to re-produce all experiments outlined in the paper are included.

Note that all model training and SHAP/IG calculation code has CLI options for setting the random seed used. To accurately
reproduce our experiments, one must set the random seed to those reported in the paper.

The code requires a number of Python libraries. Dependencies can be found in `requirements.txt`.

All models are saved as state dicts. All runnable scripts accept a number of CLI arguments which are explained through
comments and the use of `python script_name.py -h`.

- `compute_explanations.py` computes SHAP/Grad-CAM activation maps for a given baseline (i.e. non-ensemble) model.
- `compute_explanations_ensemble.py` computes the SHAP/Grad-CAM activation maps for a given explanation ensemble model.
- `ensemble.py` trains normal ensembles of U-Net/Densenet models on the MIMIC-CXR-EGD dataset
- `explanation_ensemble.py` trains our proposed explanation ensemble architecture on the MIMIC-CXR-EGD dataset
- `generate_explanation_animations.py` generates an animated GIF of Figure 4 in the Supplementary Material
- `train.py` finetunes pre-trained (on Imagenet) baseline Densenet/U-Net models on MIMIC-CXR-EGD

## MIMIC-CXR-EGD Dataset

We extensively use the [MIMIC-CXR-EGD](https://www.nature.com/articles/s41597-021-00863-5) dataset and test against some
of their baseline models using their code, which is found in the `egd_code` directory.

## Grad-CAM

For calculating Grad-CAM activation maps, we use a modified version of [this code](https://github.com/kazuto1011/grad-cam-pytorch).