
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:25:55 2022

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
from utils_old import get_vgg11_2,DOT_US_dataset,plot_roc,US_vgg11

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()

def train_US_VGG(train_us_data, train_us_label, test_us_data, test_us_label):
    train_loss = []
    vali_loss = []
    # Check GPU availablility
    # Import pretrained VGG-11 model and modified it
    # Load the pretrained model from pytorch
    vgg =get_vgg11_2()
    # training parameters and setting
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)
    # scheduler =  lr_scheduler.CosineAnnealingLR(optimizer,5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 10
    batch_size = 8
    # Generate dataset based on the data
    num_train = len(train_us_label)
    vali_split = 0.2
    vali_size = int(vali_split * num_train)
    train_size = num_train - vali_size
    dataset = US_vgg11(train_us_data, train_us_label)
    us_train_dataset, us_vali_dataset = torch.utils.data.random_split(dataset, [train_size, vali_size])
    # Generate dataloader
    train_loader = torch.utils.data.DataLoader(dataset=us_train_dataset,batch_size=batch_size,shuffle=True, drop_last=True)
    vali_loader = torch.utils.data.DataLoader(dataset=us_vali_dataset,batch_size=batch_size,shuffle=True, drop_last=True)
    # Train US vgg-11 model
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_loss = 1e10
    vgg.train(True)
    for epoch in range(num_epochs):
        loss_train, loss_val, label_sum = 0, 0, 0, 
        for (batch_idx, batch_data) in enumerate(train_loader):
            # if batch_idx % 50 == 0:
            #     print("\rTraining batch {}/{}".format(batch_idx, len(train_loader)), end='', flush=True)
            inputs, labels = batch_data
            if cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = vgg(inputs)
            _, preds = torch.max(outputs.data, 1)
            labels = labels.squeeze()
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            label_sum = label_sum + sum(labels.cpu().numpy())
            loss_train += loss.data.item()
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        scheduler.step()
        avg_loss = loss_train / train_size
        train_loss.append(avg_loss)

        # Validation part
        vgg.eval()       
        for (batch_idx, batch_data) in enumerate(vali_loader):
            # if batch_idx % 100 == 0:
            #     print("\rValidation batch {}/{}".format(batch_idx, len(vali_loader)), end='', flush=True)
            inputs, labels = batch_data
            if cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = vgg(inputs)
            _, preds = torch.max(outputs.data, 1)
            labels = labels.squeeze()
            labels = labels.long()
            loss = criterion(outputs, labels)            
            loss_val += loss.data.item()
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        avg_loss_val = loss_val / vali_size
        vali_loss.append(avg_loss_val)

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        # print("Malignant percent (%): {:.4f}".format(label_sum/train_size))
        print('-' * 10)
        print()
        # If the model improved, save it
    #     if avg_loss_val < best_loss:
    #         best_loss = avg_loss_val
    #         best_model_wts = copy.deepcopy(vgg.state_dict())
    # vgg.load_state_dict(best_model_wts)
    plt.figure
    plt.plot(train_loss, label='Training loss')
    plt.plot(vali_loss, label='Testing loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title('Loss: US')
    plt.show() 
    

    #     # print
    # Return the model for further use
    return vgg



# %%

fpr_all_us = []
tpr_all_us = []
tprs_us = []
aucs_us = []

mean_fpr = np.linspace(0, 1, 50)

open_source = sio.loadmat('../data/USmodel_dataset/open_source_normalized.mat')
patient_us_b = sio.loadmat('../data/USmodel_dataset/patient_benign_normalized.mat')
patient_us_m = sio.loadmat('../data/USmodel_dataset/patient_malignant_normalized.mat')
us_data_o, us_label_o = open_source['us_data'], open_source['label']
us_patient_o = 3000 * np.ones((len(us_label_o), 1))
us_data_b, us_label_b = patient_us_b['us_data'], patient_us_b['label']
us_data_m, us_label_m = patient_us_m['us_data'], patient_us_m['label']   
us_data = np.concatenate((us_data_b,us_data_m),axis = 0) 
us_label = np.concatenate((us_label_b,us_label_m),axis = 0) 
us_patient = np.concatenate((patient_us_b['patient_id'], patient_us_m['patient_id']))

del open_source,patient_us_b,patient_us_m,us_data_b, us_label_b,us_data_m, us_label_m

us_patient_o = 3000 * np.ones((len(us_label_o), 1))
# X_train, X_test, labels_train, labels_test = train_
for i in range(50):
    print('run = ', i)
    
    with open('../pkls/data_' + str(i) + '.pkl', 'rb') as f:
        _, _, train_us_patient, test_us_data2, test_label, test_us_patient,\
            _, _, _, _, \
            _,  _ = pickle.load(f)
 
    
    test_ind = np.in1d(us_patient, test_us_patient)
    test_us_data = us_data[test_ind,:,:,:] 
    test_label = us_label[test_ind] 
    
    train_ind = ~np.in1d(us_patient, test_us_patient)
    train_us_data = us_data[train_ind,:,:,:] 
    train_label = us_label[train_ind]
    
    train_us_data = np.concatenate((train_us_data, us_data_o), axis=0)
    train_label = np.concatenate((train_label, us_label_o), axis=0)
    vgg11 = train_US_VGG(train_us_data, train_label, test_us_data, test_label)
    # torch.save(vgg11.state_dict(), 'vgg_'+str(i)+'_os_1.pth')
    vgg11.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_us_data2,batch_size=8)

    us_test_pred = torch.empty((0,2)).to(device)
    with torch.no_grad():       
        for (batch_idx,batch) in enumerate(test_loader):
            us_test_results = vgg11(batch.to(device))
            us_test_pred = torch.cat((us_test_pred,us_test_results),0)
          
    us_test_pred = torch.softmax(us_test_pred,dim=1)[:,1].cpu().numpy()
    plot_roc(test_label.squeeze(), us_test_pred,"US receiver operating characteristic")
    
    fpr, tpr, _ = roc_curve(test_label, us_test_pred)        
    interp_tpr_us = np.interp(mean_fpr, fpr, tpr)
    interp_tpr_us[0] = 0.0
    tprs_us.append(interp_tpr_us)
    roc_auc = auc(fpr, tpr)
    aucs_us.append(roc_auc)
    
    tpr_all_us.append(tpr)
    fpr_all_us.append(fpr)
    
mean_tpr = np.mean(tprs_us, axis=0)
mean_tpr[-1] = 1.0
mean_auc_us = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs_us)
plt.plot(mean_fpr, mean_tpr, color='blue',
        label=r'US ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc_us, std_auc),
        lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

std_tpr = np.std(tprs_us, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4,
                )