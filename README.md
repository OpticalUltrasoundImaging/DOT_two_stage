# DOT_two_stage
A two-stage diagnosis approach to achieve an automated, fast, and accurate classification of breast lesions.

**by Menghao Zhang, Shuying Li, and Minghao Xue. (https://opticalultrasoundimaging.wustl.edu/)**

![This is an image](https://github.com/OpticalUltrasoundImaging/DOT_two_stage/blob/main/Figs/First_stage.png)

![This is an image](https://github.com/OpticalUltrasoundImaging/DOT_two_stage/blob/main/Figs/Second_stage.png)
## Abstract

We propose a two-stage classification strategy with deep learning. In the first stage, US images and histograms created from DOT perturbation measurements are combined to predict benign lesions. Then the non-benign suspicious lesions are passed through to the second stage, which combines US image features and DOT histogram features and 3D DOT reconstructed images for final diagnosis. The first stage alone identified 73.0% of benign cases without image reconstruction. In distinguishing between benign and malignant breast lesions in patient data, the two-stage classification approach achieved an AUC of 0.946, outperforming the diagnoses of all single-modality models, and of a single-stage classification model that combines all US images, DOT histogram and imaging features. The proposed two-stage classification strategy achieves better classification accuracy than single-modality-only models and a single-stage classification model that combines all features. It can potentially distinguish breast cancers from benign lesions in near real-time.

## Requirements
* Python: 3.7+
* torch(pytorch): 1.10.0
* torchvision: 0.11.1
* numpy: 1.21.2 
* scipy: 1.7.1
* scikit-learn: 1.3

## Details

Traing and testing files can be found in the folder Code.

**First stage**
DOT histogram: DOT_histogram_training.py
US images: train_vgg_w_opensource.py
DOT reconstruction: DOT_image_only.py

**Second stage**
Combined: DOT_hist_image_US_fusion.py

**Two stage combination**
two_stage_combine.py

## Contact

Please email Minghao Xue at m.xue@wustl.edu if you have any concern or questions.
