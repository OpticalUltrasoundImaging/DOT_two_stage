# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:02:42 2023
@author: Menghao Zhang, Shuying Li, Minghao Xue

train seperate model
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
from utils import get_vgg11_2,DOT_hist_dataset,plot_roc,plot_roc_std,plot_ave_roc,DOT_and_US_finetune_data_generate_thb_Shuying

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_DOT_histogram(pretrained,train_dataset,test_dataset,
                        lr,num_epochs,reg,earlystop,es_patience,min_epoch):
    
    if num_epochs <= 0:
        return pretrained,0
    
    model = copy.deepcopy(pretrained)
    
    train_size = len(train_dataset)
    vali_size = len(test_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    vali_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    # Learning rate decay
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,5)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,patience=3)
    # early_stopping = EarlyStopping2(patience = es_patience, stop_epoch=min_epoch, verbose=False)
    # Training epochs
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
            loss.backward(retain_graph=True)
            optimizer.step()
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
                # validation_loss /= batch_size
        # Learning rate decay
        scheduler.step()
        # if earlystop:
        #     early_stopping(epoch,validation_loss, model)
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         model.load_state_dict(torch.load('checkpoint.pt'))
        #         epoch = epoch - es_patience
        #         break
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
    plt.title('Loss: DOT histogram + DOT image + US')
    plt.show() 
    
    return model,epoch+1 


# %%
fpr_all_combine = []
tpr_all_combine = []
tprs_combine = []
aucs_combine = []

fpr_all_dot = []
tpr_all_dot = []
tprs_dot = []
aucs_dot = []

fpr_all_hist = []
tpr_all_hist = []
tprs_hist = []
aucs_hist = []

fpr_all_us = []
tpr_all_us = []
tprs_us = []
aucs_us = []

pretrained = nets.DOT_classification_all().to(device)
mean_fpr = np.linspace(0, 1, 51)

for i in range(50):
    print(i)
        
    load_name = 'pkls/first_stage_' + str(i) + '_os.pkl'
    with open(load_name, 'rb') as f:
        train_patient, test_patient, us_train_features, us_test_features, test_label, us_test_pred, dot_pred, combined_pred = pickle.load(f)

    fpr, tpr, thresh = roc_curve(test_label, us_test_pred)        
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs_us.append(interp_tpr)
    roc_auc = auc(fpr, tpr)
    aucs_us.append(roc_auc)

    # train_dataset, test_dataset, train_recon,test_recon,train_label, test_label,train_patient,test_patient,train_us_features, test_us_features\
    #     = DOT_and_US_finetune_data_generate_thb_Shuying(train_patient, test_patient, us_train_features, us_test_features)
    # # Load pretrain model
    # breakpoint()
    # dot_model = torch.load('models/dot_image_only_'+str(i)+'_new_data2.pt')
    # dot_model.eval()
    # dot_pred,dot_test_features = dot_model(test_recon.to(device))
    # dot_pred = torch.sigmoid(dot_pred).detach().cpu().numpy().squeeze()
    # _,dot_train_features = dot_model(train_recon.to(device))
    # fpr, tpr, thresh = roc_curve(test_label, dot_pred)        
    # interp_tpr = np.interp(mean_fpr, fpr, tpr)
    # interp_tpr[0] = 0.0
    # tprs_dot.append(interp_tpr)
    # roc_auc = auc(fpr, tpr)
    # aucs_dot.append(roc_auc)
    
    with open('pkls/image_only_' + str(i) + '_new_data2.pkl', 'rb') as f:
        train_patient, test_patient,train_label,test_label,prediction_train,dot_train_features,prediction_test,dot_test_features = pickle.load(f)
        
    _, _, _,_,_, _,_,_,train_us_features, test_us_features\
        = DOT_and_US_finetune_data_generate_thb_Shuying(train_patient, test_patient, us_train_features, us_test_features)

    fpr, tpr, thresh = roc_curve(test_label, prediction_test.detach().cpu().numpy())       
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs_dot.append(interp_tpr)
    roc_auc = auc(fpr, tpr)
    aucs_dot.append(roc_auc)
    
    with open('pkls/data_' + str(i) + '.pkl', 'rb') as f:
        _, _, train_us_patient, _, test_1st_label, test_us_patient,\
            _, _, _, _, \
            train_hist_data,  test_hist_data = pickle.load(f)
            
    ind = sorted(np.intersect1d(test_patient,test_us_patient, return_indices=True)[2])
    test_hist_data = test_hist_data[ind,:,:,:]
    
    ind =  sorted(np.intersect1d(train_patient,train_us_patient, return_indices=True)[2])
    train_hist_data = train_hist_data[ind,:,:,:]
    
    hist_model = nets.DOT_hist().to(device)
    hist_model.load_state_dict(torch.load('models/dot_only_'+str(i)+'_3.pth'))
    hist_model.eval()
    hist_pred,hist_test_features = hist_model(test_hist_data.to(device))
    hist_pred = torch.sigmoid(hist_pred).detach().cpu().numpy().squeeze()
    _,hist_train_features = hist_model(train_hist_data.to(device))
    fpr, tpr, thresh = roc_curve(test_label, hist_pred)        
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs_hist.append(interp_tpr)
    roc_auc = auc(fpr, tpr)
    aucs_hist.append(roc_auc)
    
    train_features = torch.cat((dot_train_features,train_us_features.to(device),hist_train_features.to(device)),1)
    test_features = torch.cat((dot_test_features,test_us_features.to(device),hist_test_features.to(device)),1)

    train_dataset = DOT_hist_dataset(train_features, train_label)
    test_dataset = DOT_hist_dataset(test_features, test_label)

    combined_model,_ = train_DOT_histogram(pretrained,train_dataset,test_dataset,1e-4,20,0,False,0,0)
    torch.save(combined_model.state_dict(), 'models/combined_all_'+str(i)+'_new_data3.pth')
    # combined_model = nets.DOT_classification_seperate().to(device)
    combined_model.eval()
    combined_pred,_ = combined_model(test_features)
    combined_pred = torch.sigmoid(combined_pred).detach().cpu().numpy()
    plot_roc(test_label.squeeze(), combined_pred,"Receiver operating characteristic")

    fpr, tpr, thresh = roc_curve(test_label, combined_pred)        
    interp_tpr_dot = np.interp(mean_fpr, fpr, tpr)
    interp_tpr_dot[0] = 0.0
    tprs_combine.append(interp_tpr_dot)
    roc_auc = auc(fpr, tpr)
    aucs_combine.append(roc_auc)
    
    tpr_all_combine.append(tpr)
    fpr_all_combine.append(fpr)
    
    with open('pkls/combined_all_' + str(i) + '_new_data3.pkl', 'wb') as f:
        pickle.dump((train_patient, test_patient,train_label, test_label, us_test_pred, hist_pred,  prediction_test.detach().cpu().numpy(), combined_pred, 
            ), f)
    
plot_ave_roc(tprs_us,mean_fpr,aucs_us,'US','navy')
plot_ave_roc(tprs_dot,mean_fpr,aucs_dot,'DOT image','green')
plot_ave_roc(tprs_hist,mean_fpr,aucs_hist,'DOT hist','orange')
plot_ave_roc(tprs_combine,mean_fpr,aucs_combine,'US + DOT histogram + DOT image','red')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title("1st stage Receiver operating characteristic")
plt.legend(loc="lower right",prop={'size': 10})
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()
