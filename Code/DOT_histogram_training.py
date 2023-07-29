
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
from utils import DOT_hist_dataset,get_simu_hist,plot_roc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_DOT_histogram(pretrained,train_dataset,test_dataset,numb,numm,
                        lr,num_epochs,earlystop,es_patience,min_epoch,lr_schedule,unbalanced):
    if num_epochs <= 0:
        return pretrained,0
    
    model = copy.deepcopy(pretrained)
    train_size = len(train_dataset)
    vali_size = len(test_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    vali_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8)
    
    if unbalanced:
        criterion = nn.BCEWithLogitsLoss(pos_weight=(torch.Tensor([numb/numm]).to(device)))

    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Learning rate decay
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,10)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,patience=3)
    # Training epochs
    # Record loss for every epoch
    train_loss = []
    vali_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0 
        for (batch_idx,batch) in enumerate(train_loader):
            x = batch[0].to(device) 
            y = batch[1].to(device) 
           
            predicts,_ = model(x)
            loss = criterion(predicts.squeeze(), y.squeeze())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print some statistics and track the loss
            running_loss += loss.item()

        # Validation loop
        # Check the validation results
        validation_loss = 0.0
        with torch.no_grad():
            for (vali_batch_idx,vali_batch) in enumerate(vali_loader):
                vali_x = vali_batch[0].to(device) 
                vali_y = vali_batch[1].to(device) 
             
                vali_pred, _ = model(vali_x)
                loss_vali = criterion(vali_pred.squeeze(), vali_y.squeeze())
                validation_loss += loss_vali.item()
        # Learning rate decay
        if lr_schedule:
            scheduler.step()
     
        # Print results every 20 epochs
        # if epoch % 1 == 0:
        #     print('Epoch [{}/{}], Training Loss: {:.8f}'
        #           .format(epoch + 1, num_epochs, running_loss/train_size))
        #     print('Epoch [{}/{}], Validation Loss: {:.8f}'
        #           .format(epoch + 1, num_epochs, validation_loss/vali_size))
        
        # Documents training and validation losses for each epochs
        train_loss.append(running_loss / train_size)
        vali_loss.append(validation_loss / vali_size)
    # print('DOT histogram training finished')
    
    plt.figure
    plt.plot(train_loss, label='Training loss')
    plt.plot(vali_loss, label='Testing loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title('Loss: DOT histogram')
    plt.show() 
    
    return model,epoch+1 
# %%

fpr_all_dot = []
tpr_all_dot = []
tprs_dot = []
aucs_dot = []

pre_trained = nets.DOT_hist().to(device)
# train_dataset, test_dataset,numb, numm = get_simu_hist()
# dot_model_pretrained,_ = train_DOT_histogram(pre_trained,train_dataset,test_dataset,numb, numm,1e-4,20,False,0,0,False,True)
# torch.save(dot_model_pretrained.state_dict(), 'dot_hist_simulation_2.pth')
dot_model_pretrained = nets.DOT_hist().to(device)
dot_model_pretrained.load_state_dict(torch.load('models/dot_hist_simulation_2.pth'))
mean_fpr = np.linspace(0, 1, 50)
# X_train, X_test, labels_train, labels_test = train_
# %%
for i in range(50):
    print('run = ', i)
    with open('pkls/data_' + str(i) + '.pkl', 'rb') as f:
        _, train_label, train_patient, _, test_label, test_patient,\
            _, _, _, _, \
             train_hist_data,  test_hist_data = pickle.load(f)

    train_dataset = DOT_hist_dataset(train_hist_data, train_label)
    test_dataset = DOT_hist_dataset(test_hist_data, test_label)
    dot_model,_ = train_DOT_histogram(dot_model_pretrained,train_dataset,test_dataset,60,60,1e-4,20,False,0,0,True,False)
    # torch.save(dot_model.state_dict(), 'dot_hist_'+str(i)+'_R1.pth')
    # dot_model.batch2.register_forward_hook(get_features('dot-fcc'))
   

    dot_model.eval()
    dot_pred,_ = dot_model(test_hist_data.to(device))
    dot_pred = dot_pred.detach().cpu().numpy()
    plot_roc(test_label,dot_pred,'DOT receiver operating characteristic')
    del dot_model
  
    fpr, tpr, _ = roc_curve(test_label, dot_pred)        
    interp_tpr_dot = np.interp(mean_fpr, fpr, tpr)
    interp_tpr_dot[0] = 0.0
    tprs_dot.append(interp_tpr_dot)
    roc_auc = auc(fpr, tpr)
    aucs_dot.append(roc_auc)    
    tpr_all_dot.append(tpr)
    fpr_all_dot.append(fpr)



mean_tpr = np.mean(tprs_dot, axis=0)
mean_tpr[-1] = 1.0
mean_auc_dot = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs_dot)
plt.plot(mean_fpr, mean_tpr, color='green',
        label=r'DOT ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc_dot, std_auc),
        lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

std_tpr = np.std(tprs_dot, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4,
                )

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right",prop={'size': 10})
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()
