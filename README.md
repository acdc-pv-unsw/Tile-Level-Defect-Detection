# Tile-Level-Defect-Detection

**Conference Abstract:**

Defect detection is a critical aspect of assuring the quality and reliability of silicon solar cells and modules. Luminescence imaging has been widely adopted as a fast method for analyzing photovoltaic devices and detecting faults. However, visual inspection of luminescence images is too slow for the expected manufacturing throughput. In this study, we propose a deep learning approach that identifies and localizes defects in electroluminescence images. Images are split into 16 tiles prior to training and treated as separate images for classification. The classified tiles provide both defect labels and their positions within the cell. We demonstrate the use of this novel approach to replace visual inspection of luminescence images in photovoltaic manufacturing lines to achieve fast and accurate defect detection.

**Conference manuscript link:** 
N/A 31/08/2021

**Description:**

Training different deep learning models to detect defective tiles

This script loops through multiple deep learning models and applies transfer
learning to detect defects on a dataset of EL images of multicrystalline cells from
fielded modules.

The dataset consists of the EL images split into 16 equivalent tiles, which were
relabelled. As the deep learning models classify the tiles between "No Anomaly",
"Crack", and "Finger Failure", the classified tiles provide the spatial information
regarding where the defect is on the cell. Therefore, this defect detection method
provides a "tile level" localisation.

Deep learning models include:

    - SqueezeNet
    - AlexNet
    - VggNet16 & 19
    - ResNet18 & 34

The transfer learning only involves redesigning the fully connected (FC) layers of the
nearal network. It takes the output of the last CNN block and through 3 FC layers
has an output of 3 neurons.

latest version: 30/08/2021

Author: Zubair Abdullah-Vetter
Email: z.abdullahvetter@unsw.edu.au

**Getting started:**

Simply clone this github and the script Tiled_Defect_Detection.py contains all the necessary code to train the different deep learning models on the tiled dataset.
Make sure to unzip the tile_imgs.zip in the Data folder, the CNNs will obtain the tile images from Data\tile_imgs
