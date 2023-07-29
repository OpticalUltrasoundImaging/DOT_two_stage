
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:31:16 2022

@author: Menghao Zhang, Shuying Li, Minghao Xue

"""

from scipy import io as sio
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import nets
import matplotlib.pyplot as plt
import time
import os
import pathlib
import copy
import statistics
from sklearn import metrics
import shutil
import random
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pickle
from utils import DOT_and_US_finetune_data_generate_thb_Shuying
# from utils import plot_roc_twoline


# def plot_roc_oneline(label,pred,title):    
#     fpr, tpr, _ = roc_curve(label, pred)        
#     roc_auc = auc(fpr, tpr)
#     plt.figure()   
#     plt.plot(fpr, tpr, color='darkorange',            
#             lw=2)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')       
#     plt.xlim([-0.05, 1.0])
#     plt.ylim([0.0, 1])
#     plt.title(title)
#     plt.text(0.6,0.2,'AUC = %0.3f' %roc_auc)
#     plt.xlabel("False positive rate")
#     plt.ylabel("True positive rate")
#     plt.show()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(pretrained,train_dataset,test_dataset,bz,
                        lr,num_epochs,reg,criterion,modality):
    
    if num_epochs <= 0:
        return pretrained,0
    
    model = copy.deepcopy(pretrained)
    
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bz, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bz)
    
    #########################################################################
    # set up for each modality
    if modality == 'us':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    
    if modality == 'hist':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,10)
    elif modality == 'us':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,10)  ## 
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif modality == 'combined_1st':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,10)
    elif modality == 'image':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)#lr_scheduler.CosineAnnealingLR(optimizer,5)#
    elif modality == 'combined_2nd':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 10)
        
        
    # Training epochs
    train_loss = []
    testing_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0 
        for (batch_idx,batch) in enumerate(train_loader):
            x = batch[0].to(device) 
            y = batch[1].to(device) 
            
            if modality == 'us':
                predicts = model(x)   

            else:
                predicts,_ = model(x)
            loss = criterion(predicts.squeeze(), y.squeeze())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()

        # testing loop
        test_loss = 0.0
        with torch.no_grad():
            for (test_batch_idx,test_batch) in enumerate(test_loader):
                test_x = test_batch[0].to(device) 
                test_y = test_batch[1].to(device) 
             
                if modality == 'us':
                    predicts = model(test_x)   
                else:
                    predicts,_ = model(test_x)
                loss_test = criterion(predicts.squeeze(), test_y.squeeze())
                test_loss += loss_test.item()
                # validation_loss /= batch_size
        # Learning rate decay
        if modality == 'hist_simu':
            pass
        else:
            scheduler.step()
     
        # Documents training and validation losses for each epochs
        train_loss.append(running_loss / train_size)
        testing_loss.append(test_loss / test_size)
    
    plt.figure
    plt.plot(train_loss, label='Training loss')
    plt.plot(testing_loss, label='Testing loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title('Loss: DOT image')
    plt.show() 
    
    return model
    
def finetune_main_DOT_classification_image_andUS_thb(run_idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load data
    load_name = 'pkls/first_stage_' + str(i) + '_os.pkl'
    with open(load_name, 'rb') as f:
        train_patient, test_patient, us_train_features, us_test_features, test_label,\
            us_test_pred, dot_pred, combined_pred = pickle.load(f)
    # Choose patient data based on the patient_id
    breakpoint()
    train_dataset, test_dataset, train_recon,test_recon,train_label, test_label,train_patient,test_patient,train_us_features, test_us_features\
        = DOT_and_US_finetune_data_generate_thb_Shuying(train_patient, test_patient, us_train_features, us_test_features)
    # Load pretrain model
    model_path = 'model/DOT_only/DOT_classification_imageOnly_simulation_0.pt'
    pretrain_model = torch.load(model_path).to(device)
    # Train model for each wavelength
    model= train_model(pretrain_model,train_dataset,test_dataset,bz=16,
                              lr=1e-3,num_epochs=10,reg=0,criterion=nn.BCEWithLogitsLoss(),modality='image')
    
    model.eval()
    prediction_train,train_features = model(train_recon.to(device))
    prediction_test,test_features = model(test_recon.to(device))
    prediction_test = torch.sigmoid(prediction_test)
    fpr, tpr, thresh = roc_curve(test_label, prediction_test.detach().cpu())        
    roc_auc = auc(fpr, tpr)
    
    print(roc_auc)
    model_path_thb = 'models/dot_image_only_' + str(run_idx) + '_new_data2.pt'
    torch.save(model, model_path_thb)
       
    with open('pkls/image_only_' + str(i) + '_new_data2.pkl', 'wb') as f:
        pickle.dump((train_patient, test_patient,train_label,test_label,prediction_train,train_features,prediction_test,test_features), f)
    return roc_auc
# Unit test
if __name__ == "__main__":
    aucs = []
    for i in range(50):
        print(i)
        # train_second_stage_main(i)
        roc_auc = finetune_main_DOT_classification_image_andUS_thb(i)
        aucs.append(roc_auc)
    print(np.mean(aucs))