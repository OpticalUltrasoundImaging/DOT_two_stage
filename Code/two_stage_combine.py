# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:50:40 2023

@author: zyf19
"""

from scipy.special import softmax
from scipy import io as sio
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pathlib
import shutil
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pickle
import torch
import nets

auc_all = []
second_aucs = []
first_aucs = []

pred_all_1st = []
pred_all_2nd = []
pred_all_final = []
labels_all = []
fn_all = []
tn_all = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

conf_matrix_first = np.array([[0, 0], [0, 0]])
# Two stage combine code
for run_idx in range(50):
    print(run_idx)
    # Based on run_idx, read the first stage and second stage results (pickle file)
    load_name = 'pkls/first_stage_' + str(run_idx) + '_os.pkl'
    with open(load_name, 'rb') as f:
        train_patient, test_patient, us_train_features, us_test_features, test_label, us_test_pred, hist_pred, _ = pickle.load(f)
    test_label = np.squeeze(test_label.numpy()) # Convert into numpy
    test_patient = np.squeeze(test_patient)
   

    # First stage threshold
    first_stage_threshold = 0.12
    # y_score_test = np.squeeze(y_score_test)
    # test_label = np.squeeze(test_label)
    score_1st = np.squeeze((hist_pred + us_test_pred)/2) #us_test_pred# 
    fpr_1st, tpr_1st, _ = roc_curve(test_label, score_1st)
    roc_auc_1st = auc(fpr_1st, tpr_1st)
    first_aucs.append(roc_auc_1st)
    
    f_pred = (score_1st > first_stage_threshold)
    
    pred_all_1st.append(f_pred)
    f_b_ind = f_pred == 0
    f_s_ind = f_pred == 1 # S means suspicious
    # Compute accuracy and confusion matrix based on this 0.35 threshold
    b_n, m_n = int(sum(test_label == 0)), int(sum(test_label == 1)) # Compute number of benign and malignant
    # t_acc =  1 - sum(abs(f_pred - test_label)) / len(test_label) # Total accuracy
    # b_acc = 1 - sum(abs(f_pred[:b_n] - test_label[:b_n])) / b_n # Benign accuracy
    # m_acc = 1 - sum(abs(f_pred[b_n:] - test_label[b_n:])) / m_n # Benign accuracy    
    # Compute confusion matrix on the first stage
    tp = sum(np.logical_and(f_pred == 1, test_label == 1))
    fn = sum(np.logical_and(f_pred == 0, test_label == 1))
    fp = sum(np.logical_and(f_pred == 1, test_label == 0))
    tn = sum(np.logical_and(f_pred == 0, test_label == 0))
    conf_matrix_first = np.array([[tn/b_n, fn/m_n], [fp/b_n, tp/m_n]]) + conf_matrix_first
    
    
    # Second stage post-processing
    load_name = 'pkls/combined_all_' + str(run_idx) + '_new_data2.pkl'
    with open(load_name, 'rb') as f:
        train_patient2, test_patient2,train_label, test_label_2, us_test_pred, hist_pred, _, combined_pred = pickle.load(f)
             
    with open('pkls/image_only_' + str(run_idx) + '_new_data2.pkl', 'rb') as f: 
        _, _,_,_,_,_,dot_pred,_ = pickle.load(f)
        
        
    dot_pred = dot_pred.detach().cpu().numpy().squeeze()   
    indices = np.where(np.isin(test_patient,test_patient2))[0]
    us_test_pred = us_test_pred[indices]
    pred_test = combined_pred# (us_test_pred + dot_pred + hist_pred)/3
    label_test =  np.squeeze(test_label_2) 
    
    # indices = np.where(~np.isin(test_patient_2, del_list))[0]
    # test_patient_2 = test_patient_2[indices]
    # label_test = label_test[indices]
    # pred_test = pred_test[indices]
    
    # Compute combined accuracy on both first and second stage
    # Cases classified as benign are classified as benign always
    confirmed_tn = tn #conf_matrix_first[0][0]
    confirmed_fn = fn #conf_matrix_first[0][1]
    fn_all.append(fn)
    tn_all.append(tn)

    added_label = np.zeros((confirmed_tn, 1))
    added_label = np.append(added_label, np.ones((confirmed_fn,1)))
    added_pred = np.zeros((confirmed_tn, 1)) #*first_stage_threshold
    added_pred = np.append(added_pred, np.zeros((confirmed_fn,1))) #
    # b_ind = np.where(score_1st < first_stage_threshold)[0]
    # added_pred = score_1st[b_ind]
    patient_1_s = test_patient[f_s_ind]
    # Find the prediction and label of those cases in the second stage
    num_patient = len(patient_1_s)
    second_label, second_pred, second_patient = np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))
    for p_idx in range(num_patient):
        p_id = patient_1_s[p_idx]
        ind = np.where(test_patient2 == p_id)[0]
        temp_pred = pred_test[ind]
        temp_label = label_test[ind]
        temp_patient = test_patient2[ind]
        second_label = np.concatenate((second_label, temp_label[:,np.newaxis]), axis=0)
        second_pred = np.concatenate((second_pred, temp_pred), axis=0) #[:,np.newaxis]
        second_patient = np.concatenate((second_patient, temp_patient), axis=0)
    # Compute ROC curve of those cases
    pred_all_2nd.append(second_pred)
    fpr, tpr, _ = roc_curve(second_label, second_pred)  
    roc_auc_1 = auc(fpr, tpr)
    # Compute added ROC curve adding the first stage cases
    final_label = np.append(second_label, added_label)
    final_pred = np.append(second_pred, added_pred)
    pred_all_final.append(final_pred)
    fpr_final, tpr_final, _ = roc_curve(final_label, final_pred)
    labels_all.append(final_label)
    roc_auc_final = auc(fpr_final, tpr_final)
    
    fpr_2nd, tpr_2nd, _ = roc_curve(label_test, pred_test)
    roc_auc_2nd = auc(fpr_2nd, tpr_2nd)
    auc_all.append(roc_auc_final)
    second_aucs.append(roc_auc_2nd)
    with open('pkls/two_stage_' + str(run_idx) + '_new2.pkl', 'wb') as f:
        pickle.dump((final_label, final_pred), f)

conf_matrix_first = conf_matrix_first/50
print(conf_matrix_first)
print(np.mean(first_aucs))

print(np.mean(second_aucs))
print(np.mean(auc_all))