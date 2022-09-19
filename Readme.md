
#
LatentGaze: Cross-Domain Gaze Estimation through Gaze-Aware Analytic Latent Code Manipulation

This code is the PyTorch implementation of LatentGaze.

To prove our code's reproducibility, we present validation of LatentGaze on MPIIFaceGaze Datsets (9,000 images).

Due to encoding and generating the images tasks are very time-consuming, we prepare the gaze estimation code while except encoder-decoder codes. Hence, we prepare validation of LatentGaze in the single-domain task.

You can find the encoder-decoder codes at https://anonymous.4open.science/r/LatentGaze/


# Datasets
Image data link : https://drive.google.com/file/d/1f_EugBGhJC9qZI1nOJq9mDI9YlnF50kG/view?usp=sharing
Images dir : './dataset/MPII_validation'

Latent code link : https://drive.google.com/file/d/1ZGP8LFd0379ZznTv-2-cPLzyoCPW7KzD/view?usp=sharing
latent codes dir : '/mpii_latent_pt_files_with_mpii/latent_pt_files'

# LatentGaze weights
weights link : https://drive.google.com/file/d/19fNm8Pwnt-w_QSUMPoeLCH82SN2ZVD8p/view?usp=sharing

dir:
'./gaze_model_best.pt'

## Create enviroments
conda env create -f LatentGaze.yaml

## Quick Run
python main.py --test_only