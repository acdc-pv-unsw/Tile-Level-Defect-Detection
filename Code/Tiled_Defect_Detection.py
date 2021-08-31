'''
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
'''
#%%-- Import packages
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import time
import random
from PIL import Image
import os
import sys
from datetime import datetime
from collections import OrderedDict

import torch
from torch.utils import data
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms, models
import GPUtil
import gc

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from tabulate import tabulate

sys.path.append("Support_Scripts")
from Pickle import SaveObj, LoadObj
from FullPathDirList import listdir_fullpath
from ImgShow import ImgShow
import uniques
from Downsizing import Downsizing
from Matplotlib_stylesheet import *

#   <subcell> Setup data directory
cwd = os.getcwd()
pkl_dir = os.path.abspath(os.path.join(cwd,"..","Pickles"))
data_dir = os.path.abspath(os.path.join(cwd,"..","Data"))
model_dir = os.path.join(pkl_dir,'Models')
results_dir = os.path.abspath(os.path.join(cwd,"..","Results"))
aresults_dir = os.path.join(results_dir,'Archive')
#   </subcell>
#%%- END Import packages

# %%-- Functions
# Class for pulling the data
class MyDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_paths, labels, transform):
        'Initialization'
        self.labels = labels
        self.list_paths = list_paths
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        path = self.list_paths[index]

        # Load data and get label
        x = Image.open(path)
        X = self.transform(x)
        y = self.labels[path]

        return X, y

# Class used to allow random rotations for data augmentations
class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

# Function for building the data loader
def BuildLoader(df,df_sect,transforms,batch_size,shuffle):
    # Build partition dict to allow sampling of the correct data and labels
    partition = {'sect': np.array(df_sect['Tile Path'])}
    labels = {}
    # Sample the correct data and labels for the set
    for label, path in zip(df['Label Int'],df['Tile Path']):
        label = int(label)
        labels[path]=label
    # Produce the set as a tensor
    Sect_set = MyDataset(partition['sect'],labels,transforms)
    # Build the loader
    return data.DataLoader(Sect_set, batch_size=batch_size,shuffle=shuffle,pin_memory=True)

# Function for splitting the data, hence building the train, valid and test sets
def Controlled_Split(df,TT_ratio,TV_ratio,random_seed):
    # Split into defect and non-defect to have an equal number in each set
    df_1 = df[df['Cell Label Int']==1] # defective cells
    df_0 = df[df['Cell Label Int']==0] # no anomaly cells
    # Obtain the indices of the first tile of each cell
    Cell_Names_1 = df_1["Cell Name"]
    Cell_Names_0 = df_0["Cell Name"]
    cell_idxs_1 = Cell_Names_1[Cell_Names_1.duplicated() == False].index.to_numpy()
    cell_idxs_0 = Cell_Names_0[Cell_Names_0.duplicated() == False].index.to_numpy()

    # Shuffle indices and split into training and test (6:2:2)
    np.random.seed(random_seed), np.random.shuffle(cell_idxs_1)
    np.random.seed(random_seed), np.random.shuffle(cell_idxs_0)
    split_1 = int(np.floor(TT_ratio * len(cell_idxs_1)))
    split_0 = int(np.floor(TT_ratio * len(cell_idxs_0)))
    c_traintot_idx_1, c_test_idx_1 = cell_idxs_1[split_1:], cell_idxs_1[:split_1]
    c_traintot_idx_0, c_test_idx_0 = cell_idxs_0[split_0:], cell_idxs_0[:split_0]
    # split train and validation indices
    split_1 = int(np.floor(TV_ratio * len(c_traintot_idx_1)))
    split_0 = int(np.floor(TV_ratio * len(c_traintot_idx_0)))
    c_train_idx_1, c_valid_idx_1 = c_traintot_idx_1[split_1:], c_traintot_idx_1[:split_1]
    c_train_idx_0, c_valid_idx_0 = c_traintot_idx_0[split_0:], c_traintot_idx_0[:split_0]

    # Combine to form full train and test cell indices
    c_train_idx = np.append(c_train_idx_1,c_train_idx_0)
    c_valid_idx = np.append(c_valid_idx_1,c_valid_idx_0)
    c_test_idx = np.append(c_test_idx_1,c_test_idx_0)
    np.random.seed(random_seed), np.random.shuffle(c_train_idx)
    np.random.seed(random_seed), np.random.shuffle(c_valid_idx)
    np.random.seed(random_seed), np.random.shuffle(c_test_idx)

    print('Total number of defect train cells = {0}, no anomaly train cells = {1}'.format(len(c_train_idx_1),len(c_train_idx_0)))
    print('Total number of defect validation cells = {0}, good validation cells = {1}'.format(len(c_valid_idx_1),len(c_valid_idx_0)))
    print('Total number of defect test cells = {0}, good test cells = {1}'.format(len(c_test_idx_1),len(c_test_idx_0)))

    # collecting tiles as separate dataframes
    tile_train_idx = []
    tile_valid_idx = []
    tile_test_idx = []
    for i, row in df.iterrows():
        name = row['Cell Name']
        if name in np.unique(df.loc[c_train_idx]["Cell Name"]):
            tile_train_idx.append(i)
        elif name in np.unique(df.loc[c_test_idx]["Cell Name"]):
            tile_test_idx.append(i)
        elif name in np.unique(df.loc[c_valid_idx]["Cell Name"]):
            tile_valid_idx.append(i)
    df_train = df.loc[tile_train_idx]
    df_train = Downsizing(df_train,random_seed=random_seed).sample(frac=1).reset_index(drop=True) # downscale and shuffle
    df_valid = df.loc[tile_valid_idx]
    df_valid = Downsizing(df_valid,random_seed=random_seed).sample(frac=1).reset_index(drop=True) # downscale and shuffle
    df_test = df.loc[tile_test_idx]

    return df_train, df_valid, df_test

# Function for choosing model function with transfer learning
def Model_Selection(model_choice):
    '''Model choices available'''
    model_list = "AlexNet, SqueezeNet, VggNet16, VggNet19, ResNet18, ResNet34"

    # Load in chosen model
    if model_choice == "SqueezeNet":
        model = models.squeezenet1_0(pretrained=False)
        num_classes = 3
        # change the last conv2d layer
        model.classifier._modules["1"] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        # change the internal num_classes variable rather than redefining the forward pass
        model.num_classes = num_classes
        print('Loading untrained squeezenet')

    elif model_choice == "VggNet16":
            model = models.vgg16(pretrained=True)

            # Freeze parameters so we don't backprop through them
            for param in model.features[0:28].parameters():
                param.requires_grad = False
            classifierVgg = nn.Sequential(OrderedDict([
                                      ('fc1', nn.Linear(25088, 1024)),
                                      ('relu', nn.ReLU()),
                                      ('dropout', nn.Dropout(p=0.5)),
                                      ('fc2', nn.Linear(1024, 512)),
                                      ('relu', nn.ReLU()),
                                      ('dropout', nn.Dropout(p=0.5)),
                                      ('fc3', nn.Linear(512, 3)),
                                      ('output', nn.Softmax(dim=1))
                                      ]))
            model.classifier = classifierVgg
            print('Loading pretrained VggNet16')

    elif model_choice == "VggNet19":
        model = models.vgg19(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.features[0:34].parameters():
            param.requires_grad = False
        classifierVgg = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, 1024)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.5)),
                                  ('fc2', nn.Linear(1024, 512)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.5)),
                                  ('fc3', nn.Linear(512, 3)),
                                  ('output', nn.Softmax(dim=1))
                                  ]))
        model.classifier = classifierVgg
        print('Loading pretrained VggNet19')

    elif model_choice == "AlexNet":
        model = models.alexnet(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.features[0:10].parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(9216, 1024)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.5)),
                                  ('fc2', nn.Linear(1024, 512)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.5)),
                                  ('fc3', nn.Linear(512, 3)),
                                  ('output', nn.Softmax(dim=1))
                                  ]))
        model.classifier = classifier
        print('Loading pretrained AlexNet')

    elif model_choice == "ResNet18":
        model = models.resnet18(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze last conv block
        for param in model.layer4.parameters():
            param.requires_grad = True
        classifierRes = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(512, 256)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.5)),
                                  ('fc2', nn.Linear(256, 64)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.5)),
                                  ('fc3', nn.Linear(64, 3)),
                                  ('output', nn.Softmax(dim=1))
                                  ]))
        model.fc = classifierRes
        print('Loading pretrained ResNet18')

    elif model_choice == "ResNet34":
        model = models.resnet34(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze last conv block
        for param in model.layer4.parameters():
            param.requires_grad = True
        classifierRes = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(512, 256)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.5)),
                                  ('fc2', nn.Linear(256, 64)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.5)),
                                  ('fc3', nn.Linear(64, 3)),
                                  ('output', nn.Softmax(dim=1))
                                  ]))
        model.fc = classifierRes
        print('Loading pretrained ResNet34')

    else:
        print("Incorrect model choice\n")
        print("Available models = {}".format(model_list))
        return
    return model

# Function for running a single training instance of a model
# i.e. running through the epochs etc
# saves the best model based on improved validation score
def training_instance(n_epochs,criterion,optimizer):
    # intialise and loss arrays
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    valid_loss_min = np.Inf # track change in validation loss

    for epoch in range(n_epochs):
        print('GPU memory at epoch {0} :{1:3.0f}%'.format(epoch,GPUtil.getGPUs()[0].memoryUtil*100))

        train_loss = 0.0
        valid_loss = 0.0
        accuracy = 0
        # Start timer
        start = time.time()
        model.train()
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # Move data (input) and target(label) tensors to the default device
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            output, loss, data, target = output.detach(), loss.detach(), data.detach(), target.detach()
            train_loss += loss.item()*data.size(0)
            torch.cuda.empty_cache()
            #print('GPU memory:{:3.0f}%'.format(GPUtil.getGPUs()[0].memoryUtil*100))


        with torch.no_grad():
            model.eval()
            for data, target in valid_loader:
                # Move data (input) and target(label) tensors to the default device
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                output, loss, data, target = output.detach(), loss.detach(), data.detach(), target.detach()
                valid_loss += loss.item()*data.size(0)
                # Get the highest classification probabilities for each data point
                top_p, top_class = output.topk(1, dim=1)
                # cross check with labels to check if they are correct
                equals = top_class == target.view(*top_class.shape)
                # Calculate accuracy
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                torch.cuda.empty_cache()


        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        train_losses.append(train_loss)
        valid_loss = valid_loss/len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        valid_accuracy = accuracy/len(valid_loader)
        valid_accuracies.append(valid_accuracy)

        # print training/validation statistics
        print("Epoch: {}/{}.. ".format(epoch+1, n_epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Valid Loss: {:.3f}.. ".format(valid_loss),
              "Valid Accuracy: {:.3f}".format(valid_accuracy),
              "Time/epoch: {:.3f}s".format(time.time() - start))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), os.path.join(model_dir,model_choice+model_name))
            valid_loss_min = valid_loss
    return train_losses,valid_losses,valid_accuracies

# Function to evaluate the final model
def Model_Eval(model,loader):
    model.to(device)
    y_true = []
    y_pred = []
    probabilities = []
    # print('GPU memory at testing:{:3.0f}%'.format(GPUtil.getGPUs()[0].memoryUtil*100))
    # push test data through trained model
    with torch.no_grad():
        model.eval()
        for data, target in loader:
            # Move data (input) and target(label) tensors to the default device
            data = data.to(device)
            ps = model(data)
            ps, data = ps.cpu(), data.detach()
            top_p, top_class = ps.topk(1, dim=1)
            y_true.extend(target.numpy())
            y_pred.extend(top_class.numpy())
            probabilities.extend(ps.numpy())

            torch.cuda.empty_cache()

    # F1score and accuracy score
    F1score = (f1_score(y_true, y_pred,average='macro')*100).astype(int)
    Accuracy = (accuracy_score(y_true,y_pred)*100).astype(int)

    return probabilities, F1score, Accuracy, y_true, y_pred

# Function for collating tile classification to cell level classification
def Tile_Locate(df_test):
    # Initialise counting and cell names
    # u,count = np.unique(df_test['Cell Label Int'],return_counts=True)
    TPCount = 0
    FNCount = 0
    TNCount = 0
    FPCount = 0
    FNCells = []
    FPCells = []
    # Initialise new y_test and y_pred list
    new_y_pred = []
    Cell_Label_predict = []
    for Cell in np.unique(df_test['Cell Name']): # Cell names of WHOLE cells
        df_Cell = df_test[df_test['Cell Name'] == Cell]
        if not df_Cell.empty:
            df_True = df_Cell[df_Cell['Label Int']==df_Cell['Label pr']] # Correctly predicted tiles
            df_False = df_Cell[df_Cell['Label Int']!=df_Cell['Label pr']] # Incorrectly predicted tiles
            if df_Cell['Label pr'].isin([1,2]).any().any(): # defect tile exists
                if df_True['Label Int'].isin([1,2]).any().any(): # defect cells with atleast 1 correct defect tile
                    TPCount += 1
                    new_y_pred.append(1)
                    Cell_Label_predict.extend(np.ones(16,np.uint8))
                else:
                    FPCount += 1
                    FPCells.append(Cell)
                    new_y_pred.append(1)
                    Cell_Label_predict.extend(np.ones(16,np.uint8))
            elif ~df_Cell['Label pr'].isin([1,2]).any().any(): # No Anomaly Cells
                N_tiles = len(df_True['Label Int']==0)
                if N_tiles>1: # No anomaly cells with atleast 1 correct tiles
                    TNCount += 1
                    new_y_pred.append(0)
                    Cell_Label_predict.extend(np.zeros(16,np.uint8))
                else:
                    FNCount += 1
                    FNCells.append(Cell)
                    new_y_pred.append(0)
                    Cell_Label_predict.extend(np.zeros(16,np.uint8))

    new_y_test = df_test[df_test["Cell Name"].duplicated() == False]['Cell Label Int']
    # F1score
    F1score = (f1_score(new_y_test, new_y_pred,average='macro')*100).astype(int)
    # Accuracy
    Accuracy = (accuracy_score(new_y_test, new_y_pred)*100).astype(int)
    # Store cell predictions in df_test
    df_test['Cell Label pr'] = Cell_Label_predict
    return df_test, F1score, Accuracy

# Function for applying localisation by colouring the different class predictions
# Images are saved with tiles separate and whole cells with concatenated coloured tiles
def Colour_tiles(df_test):
    colours = []
    # Remember RGB is actually BGR here
    for label, predict in zip(df_test['Label Int'],df_test['Label pr']):
        if ((predict==0) & (label==0)):
            colours.append([0,0,0]) # TN = black
        elif ((predict==1) & (label==1)):
            colours.append([0,255,255]) # TP finger failure = yellow
        elif ((predict==2) & (label==2)):
            colours.append([0,255,0]) # TP crack = green
        elif ((predict==0) & (label==1)):
            colours.append([42,39,255]) # FN = red
        elif ((predict==0) & (label==2)):
            colours.append([42,39,255]) # FN = red
        elif ((predict==1) & (label==0)):
            colours.append([255,0,255]) # FP = pink
        elif ((predict==2) & (label==0)):
            colours.append([255,0,255]) # FP = pink
        elif ((predict==1) & (label==2)):
            colours.append([0,165,255]) # mixed crack and finger = orange
        elif ((predict==2) & (label==1)):
            colours.append([0,165,255]) # mixed crack and finger = orange
        else:
            print("no colour "+str(predict)+','+str(label))
    # Attach the border colour to the dataframe
    df_test['Colour'] = colours
    # Print borders on the tiles to signify the tile level localisation
    stitching_dir = os.path.join(results_dir,'Localised Images')
    new_dir = os.path.join(stitching_dir,model_choice + "_" +model_name[:-3])
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    bb = 2 # border size
    for Cell in np.unique(df_test['Cell Name']):
        df_Cell = df_test[df_test['Cell Name'] == Cell].reset_index()
        cell_name = Cell.split('.')[0] # Cell name
        for index,tile in df_Cell.iterrows():
            tile_img = cv2.imread(tile['Tile Path'])
            # normaliser = np.zeros((64, 64))
            # tile_img = cv2.normalize(tile_img,  normaliser, 25, 175, cv2.NORM_MINMAX) # normalise pixel range (0-255)
            tile_border = cv2.copyMakeBorder(tile_img, bb, bb, bb, bb, cv2.BORDER_CONSTANT, value=tile['Colour']) #draw border
            # save bordered tiles to their folders
            new_cell_dir = os.path.join(new_dir,cell_name)
            if not os.path.exists(new_cell_dir):
                os.makedirs(new_cell_dir)
            new_tile_path = os.path.join(new_cell_dir,tile['Tile Name'])
            cv2.imwrite(new_tile_path,tile_border)
        # Call all the files of the bordered tiles and merge to form original cell image
        filenames = listdir_fullpath(new_cell_dir)

        images_C1 = [cv2.imread(img) for img in filenames[:4]]
        images_C2 = [cv2.imread(img) for img in filenames[4:8]]
        images_C3 = [cv2.imread(img) for img in filenames[8:12]]
        images_C4 = [cv2.imread(img) for img in filenames[12:16]]

        C1 = np.concatenate(images_C1, axis=0)
        C2 = np.concatenate(images_C2, axis=0)
        C3 = np.concatenate(images_C3, axis=0)
        C4 = np.concatenate(images_C4, axis=0)

        Whole = np.concatenate((C1,C2,C3,C4),axis=1)
        #ImgShow(Whole)
        tiled_image_whole = os.path.join(new_dir,cell_name+'.png')
        cv2.imwrite(tiled_image_whole,Whole)

    return df_test
# %%- END Functions

#%%-- Training Section
'''
Important information here is the models of choice and number of training instances
per model. The full list is the commented section below.

Multiple instances are used to provide some statistics on the test set results

Keep in mind the training loop spans across all the different subcells
'''
# All_models = ["AlexNet, Squeeze, VGG16, VGG19, ResNet18, ResNet34"]
All_models = ["AlexNet","VggNet19"]
no_instances = 3

#       <subcell> Initisialise training loop
# Setup total results collection
total_results_csv = datetime.today().strftime('%Y-%m-%d')+"_All Results.csv"
total_results_path = os.path.join(results_dir,total_results_csv)
if not os.path.exists(total_results_path):
    Columns = ['Datetime','Model','F1 test','Acc test','Time',
               'AP','Cell F1','Cell Acc','F1 train','Acc train',
               'F1 valid','Acc valid']
    total_results_df = pd.DataFrame()
    total_results_df.to_csv(total_results_path,index=False)
else:
    total_results_df = pd.read_csv(total_results_path)

# check if CUDA is available (utilise GPU)
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    device = torch.device("cpu")
else:
    print('CUDA is available!  Training on GPU ...')
    torch.cuda.empty_cache()
    device = torch.device("cuda")

# Loop over the entire training and testing loop multiple times
for model_choice in All_models:
    for i in range(no_instances):

        start_time_total = time.time()
        # datetime for saving
        dd = datetime.today().strftime('\%Y-%m-%d-%H%M')
        print('##################### New Instance - {} ##################'.format(dd[1:]))
        # Set random seed (used as unique identifier)
        random_seed = np.random.randint(0,100)
        np.random.seed(random_seed)

        '''Or used saved model random seed'''
        # saved_model = 'Squeeze_2021-05-21-1824_79.pt'
        # saved_model = 'Squeeze_2021-05-21-1752_9.pt'
        # saved_model = 'Squeeze_2021-05-21-1803_41.pt'
        # model_choice = saved_model.split('_')[0]
        # model_choice = "SqueezeNet"
        # model_name = '_'.join(saved_model.split('_')[1:])
        # dd = saved_model.split('_')[1]
        # random_seed = int(saved_model.split('_')[-1][:-3])

        # Select the right files and define file names, no busbar = _NoBusbar file names
        Remove_Busbars = True
        if Remove_Busbars:
            busbar_status = '_NoBusbar'
        else:
            busbar_status = ''
        # Load the data and build data loaders
        tile_dir = os.path.join(data_dir,'tiles_imgs') # Dir of tiles
        tile_paths = FullPathDirList(tile_dir) # grabs the whole path of the tile images

        df = pd.read_csv(os.path.join(data_dir,"Tiles_labels_new"+busbar_status+".csv")) # # DF of tiles with labels
        df['Tile Path'] = tile_paths # change the path list such that the images are obtainable
        u, counts = np.unique(df['Label'],return_counts=True)
        TT_ratio = 0.20 # Train test ratio
        TV_ratio = 0.25 # Train valid ratio (becomes 6:2:2)
        # Load train, valid and test set
        df_train, df_valid, df_test = Controlled_Split(df,TT_ratio,TV_ratio,random_seed)
        # define transforms
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               MyRotationTransform(angles=[0,90,180,-90]),
                                               transforms.RandomResizedCrop(size=64),
                                               transforms.Grayscale(3),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        test_transform = transforms.Compose([transforms.Grayscale(3),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        # Rotate 180 degrees to see if the model still thinks its a defect
        # test_transform_2 = transforms.Compose([transforms.Grayscale(3),
        #                                       MyRotationTransform(angles=[180]),
        #                                       transforms.ToTensor(),
        #                                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        # Build loaders for train, valid, test
        train_loader = BuildLoader(df,df_train,train_transform,16,True)
        valid_loader = BuildLoader(df,df_valid,test_transform,16,True)
        test_loader = BuildLoader(df,df_test,test_transform,16,False)
        test_loader2 = BuildLoader(df,df_test,test_transform_2,16,False)
        # Eval versions of train and validation
        train_loader_noeval = BuildLoader(df,df_train,test_transform,16,False)
        valid_loader_noeval = BuildLoader(df,df_valid,test_transform,16,False)
#       </subcell>

#       <subcell> Conduct a training instance
        # Transfer Learning
        model_name = "_"+dd[1:]+"_"+str(random_seed)+".pt" # for saving

        model = Model_Selection(model_choice)
        # model = Model_Selection("SqueezeNet")
        # send to GPU
        before = (GPUtil.getGPUs()[0].memoryUtil*100)
        model.to(device)
        print('Model memory requirement ={:3.0f}%'.format(GPUtil.getGPUs()[0].memoryUtil*100 - before))

        # Define loss criteria and optimiser
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
        n_epochs = 250

        # Conduct a single training instance
        train_losses,valid_losses,valid_accuracies = training_instance(n_epochs,criterion,optimizer)
        print("Time taken for all training: {:.3f}s".format(time.time() - start_time_total))
        # plot learning curve
        for i in range(1):
            fig1 = plt.figure()
            ax = plt.subplot()
            # ax.set_title('Learning Curve - {}'.format(model_choice),fontsize=34)
            ax.plot(range(n_epochs),train_losses,label='Train',c='b')
            ax.plot(range(n_epochs),valid_losses,label='Valid',c='orange')
            ax.set_xlabel('Number of Epochs',fontsize=30,labelpad=15)
            ax.set_ylabel('Cross Entropy Loss',fontsize=30,labelpad=15)
            yaxis2 = ax.twinx()
            yaxis2.plot(range(n_epochs),valid_accuracies,label="Valid Accuracy",c='r')
            yaxis2.set_ylabel('Accuracy',fontsize=30,labelpad=15)
            yaxis2.set_ylim(0,1)
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = yaxis2.get_legend_handles_labels()

            lines = lines_1 + lines_2
            labels = labels_1 + labels_2

            ax.legend(lines, labels, fontsize=22)
        # save figures
        fig1_name = os.path.join(aresults_dir, model_choice + model_name[:-3] + '_learning.png')
        plt.savefig(fig1_name)
        plt.close(fig1)

        # delete all variables to clear up GPU
        print('GPU memory at the end of training:{0:1.0f}%'.format(GPUtil.getGPUs()[0].memoryUtil*100))
        model = model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        print('GPU memory after deleting model:{0:1.0f}%'.format(GPUtil.getGPUs()[0].memoryUtil*100))

#           <ssubcell> archive saves
        # Collect learning results
        dict_learning = {"Epoch":n_epochs,"Train loss":train_losses,"Valid loss":valid_losses,"Valid Accuracy":valid_accuracies}
        df_learning = pd.DataFrame(dict_learning)
        # save archive of test results
        arch_csv_name = model_choice + model_name[:-3] +'_learning.csv'
        arch_learning = os.path.join(aresults_dir, arch_csv_name)
        df_learning.to_csv(arch_learning,index=False)
#           </ssubcell>
#       </subcell> END Training instance

#       <subcell> Begin evaluating on test set
        # Now run through all unseen data (test) to compute F1score of model
        """Use this code if you are continuing from above"""
        state_dict = torch.load(os.path.join(model_dir,model_choice+model_name))
        model.load_state_dict(state_dict)
        """Otherwise place name of the model here"""
        # model.load_state_dict(torch.load(os.path.join(model_dir,saved_model))) # if you wanted to load in your own model

        # Start evaluation
        start_time_testing = time.time()
        probabilities, test_f1, test_acc, y_true, y_pred = Model_Eval(model,test_loader)
        # probabilities, test_f1, test_acc, y_true, y_pred = Model_Eval(model,test_loader2) # rotated 180 degrees
#           <ssubcell> Load in past results here
        """Can also load in past results"""
        # csv_name = "Squeeze_2021-05-21-1803_41_results.csv"
        # df_test = pd.read_csv(os.path.join(aresults_dir, csv_name))
        # y_true = df_test['Label Int']
        # y_pred = df_test['Label pr']
        # test_f1 = (f1_score(y_true, y_pred,average='macro')*100).astype(int)
        # test_acc = (accuracy_score(y_true,y_pred)*100).astype(int)
        # model_choice = "SqueezeNet"
#           </ssubcell>

        #Calculate time taken to classify tiles in test set
        test_set_time = (time.time() - start_time_testing)/len(test_loader)
        print("Time taken for tile classification: {:.3f}s/tile".format(test_set_time))
        # Remove the beginning of the colormap
        interval = np.hstack([np.linspace(0.15, 0.4), np.linspace(0.4, 1)])
        colors = plt.cm.Purples(interval)
        cmap = LinearSegmentedColormap.from_list('name', colors)
        # Compute and plot confusion matrix
        CM_tile = confusion_matrix(y_true,y_pred)
        CM_tile[0,0] = 50
        labels = ['No Anomaly','Finger Failure','Crack']
        for i in range(1):
            fig2 = plt.figure()
            ax = plt.subplot()
            sns.heatmap(CM_tile, annot=True,annot_kws={'size': 28}, ax = ax,cmap=cmap, fmt='d') #annot=True to annotate cells
            # labels, title and ticks
            # ax.invert_yaxis()
            # ax.xaxis.set_label_position('top')
            # ax.xaxis.tick_top()
            ax.set_xlabel('Predicted labels',fontsize=28,labelpad=15)
            ax.set_ylabel('True labels',fontsize=28,labelpad=15)
            ax.set_title('{0} - Tile level classification (test set):\nF1 = {1}%, Accuracy = {2}%'.format(model_choice,test_f1,test_acc),fontsize=28)
            ax.set_xticklabels(labels,fontsize=25)
            ax.set_yticklabels(labels, va='center',fontsize=25)

        # Print CM to output
        cm_list = CM_tile.tolist()
        cm_list[0].insert(0,'No Anomaly')
        cm_list[1].insert(0,'Finger failure')
        cm_list[2].insert(0,'Crack')
        print("### Confusion Matrix (testset) - {} ###\n".format(model_choice))
        print(tabulate(cm_list,headers=['True/Pred','No Anomaly','Finger Failure','Crack']))

        # Save figure
        fig2_name = os.path.join(aresults_dir , model_choice + model_name[:-3] + '_tile CM.png')
        plt.savefig(fig2_name)
        plt.close(fig2)

#           <ssubcell> archive saves
        df_test['Label pr'] = np.array(y_pred).flatten() # Collect results
        # save archive of test results
        arch_csv_name = model_choice + model_name[:-3] +'_results.csv'
        arch_results = os.path.join(aresults_dir, arch_csv_name)
        df_test.to_csv(arch_results,index=False)
#           </ssubcell>

#           <ssubcell> Compute and plot precision-recall curve
        # convert to usable dtypes via one-vs-all scheme
        proba = np.asarray(probabilities)
        y_true = label_binarize(y_true,classes=[0,1,2])
        precision = dict()
        recall = dict()
        threshold = dict()
        AP = np.round(average_precision_score(y_true,proba),2)
        # plot precision recall curve
        for i in range(1):
            fig3 = plt.figure(figsize=(8,6))
            ax = plt.subplot()
            n_classes = 3
            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                                    proba[:, i])
                ax.plot(recall[i], precision[i], '.', label='class {}'.format(i))

            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.legend(loc="best")
            ax.set_title("{0} - precision-recall curve: AP = {1}".format(model_choice,AP),fontsize=16)

        fig3_name = os.path.join(aresults_dir, model_choice + model_name[:-3] + '_PRC.png')
        plt.savefig(fig3_name)
        plt.close(fig3)
#           </ssubcell> END precision recall curve
#       </subcell> END unseen test data

#       <subcell> Tile level localisation
        csv_name = os.path.join(aresults_dir, arch_csv_name)
        # csv_name = os.path.join(aresults_dir,'Squeeze__2021-05-21-1813_89_results.csv')
        df_test = pd.read_csv(csv_name)

        df_test, F1score, Accuracy = Tile_Locate(df_test)
#           <ssubcell> Load in past results
        """Can also load in past results"""
        # csv_name = "Squeeze_2021-05-21-1803_41_results.csv"
        # # csv_name = "Squeeze_2021-05-21-1813_89_results.csv"
        # # csv_name = "Squeeze_2021-05-21-1824_79_results.csv"
        # # csv_name = "ResNet18_2021-05-23-2049_71_results.csv"
        # df_test = pd.read_csv(os.path.join(aresults_dir, csv_name))
        # y_true = df_test[df_test["Cell Name"].duplicated() == False]['Cell Label Int']
        # y_pred = df_test[df_test["Cell Name"].duplicated() == False]['Cell Label pr']
        # F1score = (f1_score(y_true, y_pred,average='macro')*100).astype(int)
        # Accuracy = (accuracy_score(y_true,y_pred)*100).astype(int)
        # model_name = '_'.join(csv_name.split('_')[1:3])+".pt"
        # model_choice = csv_name.split('_')[0]
        # model_choice = "SqueezeNet"
        # df_test = df_test.drop("Colour",axis=1)
#           </ssubcell>
        # Remove the beginning of the colormap
        interval = np.hstack([np.linspace(0.15, 0.4), np.linspace(0.4, 1)])
        colors = plt.cm.Purples(interval)
        cmap = LinearSegmentedColormap.from_list('name', colors)
        cell_test = df_test[df_test["Cell Name"].duplicated() == False]['Cell Label Int']
        cell_pr = df_test[df_test["Cell Name"].duplicated() == False]['Cell Label pr']
        CM = confusion_matrix(cell_test,cell_pr)
        labels = ['No Anomaly','Defect']
        for i in range(1):
            fig4 = plt.figure()
            ax = plt.subplot()
            sns.heatmap(CM, annot=True,annot_kws={'size': 28}, ax = ax,cmap=cmap) #annot=True to annotate cells
            # labels, title and ticks
            # ax.invert_yaxis()
            ax.set_xlabel('Predicted labels',fontsize=28,labelpad=15)
            ax.set_ylabel('True labels',fontsize=28,labelpad=15)
            ax.set_title('{0} - Cell level classification (test set):\nF1 = {1}%, Accuracy = {2}%'.format(model_choice,F1score,Accuracy),fontsize=28)
            ax.set_xticklabels(labels,fontsize=25)
            ax.set_yticklabels(labels, va='center',fontsize=25)
            # use matplotlib.colorbar.Colorbar object
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=25)
        # save figures
        fig4_name = os.path.join(aresults_dir,model_choice + model_name[:-3] + '_cell CM.png')
        plt.savefig(fig4_name)
        plt.close(fig4)

#           <ssubcell> archive saves
        # save archive of test results
        arch_csv_name = model_choice + model_name[:-3] + '_results.csv'
        arch_results = os.path.join(aresults_dir, arch_csv_name)
        df_test.to_csv(arch_results,index=False)
#           </ssubcell>
#       </subcell> END Tile level localisation

#       <subcell> Investigate predictions and colour tiles
        csv_name = os.path.join(aresults_dir, arch_csv_name)
        df_test = pd.read_csv(csv_name)
        df_test = Colour_tiles(df_test)
#           <ssubcell> archive saves
        # save archive of test results
        arch_csv_name = model_choice + model_name[:-3] + '_results.csv'
        arch_results = os.path.join(aresults_dir, arch_csv_name)
        df_test.to_csv(arch_results,index=False)
#           </ssubcell>

#       </subcell> END investigate predictions and colour tiles

#       <subcell> Run final model on training and validation sets (to compare to test set eval)

        """Use this code if you are continuing from above"""
        state_dict = torch.load(os.path.join(model_dir,model_choice+model_name))
        model.load_state_dict(state_dict)
        """Otherwise place name of the model here"""
        # model.load_state_dict(torch.load(os.path.join(model_dir,saved_model))) # if you wanted to load in your own model

        # evaluate the model on the trainig data
        probabilities, train_f1, train_acc, y_true, y_pred = Model_Eval(model,train_loader_noeval)

        # Compute and plot confusion matrix
        CM = confusion_matrix(y_true,y_pred)
        labels = ['No Anomaly','Finger Failure','Crack']
        for i in range(1):
            fig5 = plt.figure()
            ax = plt.subplot()
            sns.heatmap(CM, annot=True, ax = ax,cmap='GnBu', fmt='d') #annot=True to annotate cells
            # labels, title and ticks
            ax.set_xlabel('Predicted labels',labelpad=15)
            ax.set_ylabel('True labels',labelpad=15)
            ax.set_title('{0} - Tile level classification (train set):\nF1 = {1}%, Acc = {2}%'.format(model_choice,train_f1,train_acc),fontsize=16)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels, va='center')

        fig5_name = os.path.join(aresults_dir , model_choice + model_name[:-3] + '_tile CM trainset.png')
        plt.savefig(fig5_name)
#           <ssubcell> archive saves
        df_train['Label pr'] = np.array(y_pred).flatten() # Collect results
        # save archive of test results
        arch_csv_name = model_choice + model_name[:-3] + '_results_trainset.csv'
        arch_results = os.path.join(aresults_dir, arch_csv_name)
        df_test.to_csv(arch_results,index=False)
#           </ssubcell>

        probabilities, valid_f1, valid_acc, y_true, y_pred = Model_Eval(model,valid_loader_noeval)

        # Compute and plot confusion matrix
        CM = confusion_matrix(y_true,y_pred)
        labels = ['No Anomaly','Finger Failure','Crack']
        for i in range(1):
            fig6 = plt.figure()
            ax = plt.subplot()
            sns.heatmap(CM, annot=True, ax = ax,cmap='GnBu', fmt='d') #annot=True to annotate cells
            # labels, title and ticks
            ax.set_xlabel('Predicted labels',labelpad=15)
            ax.set_ylabel('True labels',labelpad=15)
            ax.set_title('{0} - Tile level classification (valid set):\nF1 = {1}%, Acc = {2}%'.format(model_choice,valid_f1,valid_acc),fontsize=16)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels, va='center')
        fig6_name = os.path.join(aresults_dir , model_choice + model_name[:-3] + '_tile CM validset.png')
        plt.savefig(fig5_name)
        plt.close(fig6)
#           <ssubcell> archive saves
        df_valid['Label pr'] = np.array(y_pred).flatten() # Collect results
        # save archive of test results
        arch_csv_name = model_choice + model_name[:-3] + '_results_validset.csv'
        arch_results = os.path.join(aresults_dir, arch_csv_name)
        df_test.to_csv(arch_results,index=False)
#           </ssubcell>
        # update results dataframe
        Instance_Results = {'Datetime':dd[1:],'Model':model_choice,'F1 test':test_f1,'Acc test':test_acc,'Time':test_set_time,
                            'AP':AP,'Cell F1':F1score,'Cell Acc':Accuracy,'F1 train':train_f1,'Acc train':train_acc,
                            'F1 valid':valid_f1,'Acc valid':valid_acc}
        total_results_df = total_results_df.append(Instance_Results,ignore_index=True)
        total_results_df.to_csv(total_results_path,index=False)

        print("Time taken for full loop: {:.3f}s".format(time.time() - start_time_total))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print('Model memory usage at end of an instance ={:3.0f}%'.format(GPUtil.getGPUs()[0].memoryUtil*100))
        plt.close('all') # close all figures

    # calculate and store average results
    # model_results_df = total_results_df[total_results_df['Model']==model_choice]
    # avrg_results = model_results_df.mean(numeric_only=True,axis=0)
#       </subcell>

print("End of all training instances and all results are saved")
#%%- END of all instances - Training complete

#%%-- Investigate total results
# Load in results
# csv = "2021-05-21_All Results_v2.csv"
dff = pd.read_csv(os.path.join(results_dir,csv))

for i in range(1):
    g = sns.boxplot(x=dff["Model"], y=dff["Cell F1-Score"],palette="tab10")
    g.set_xticklabels(rotation=30,labels=np.unique(dff['Model']))
    # g.set_yticklabels(g.get_yticks(), size = 22)
    g.set_xlabel('Model',fontsize=30)
    g.set_ylabel('Cell F1-score',fontsize=30)
    g.set_title("F1-scores of cell level classification",fontsize=34)

# produce boxplots of the total data
h = sns.boxplot( x=dff["Model"], y=dff["Cell Accuracy"] )
h.set_xticklabels(rotation=30,labels=np.unique(dff['Model']))
h.set_title("Box and whisker plot of accuracies",fontsize=22)

ii = sns.boxplot( x=dff["Model"], y=dff["Tile Accuracy"] )
ii.set_xticklabels(rotation=30,labels=np.unique(dff['Model']))
ii.set_title("Box and whisker plot of accuracies for different architectures",fontsize=22)

ii = sns.boxplot( x=dff["Model"], y=dff["Tile F1-Score"] )
ii.set_xticklabels(rotation=30,labels=np.unique(dff['Model']))
ii.set_title("Box and whisker plot of accuracies for different architectures",fontsize=22)
#%%-
