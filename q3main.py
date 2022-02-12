import numpy as np
import random as rnd
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import math

def normalize(data):
  min = np.amin(data)
  max = np.amax(data)
  for index, item in enumerate(data):
    data[index] = (item - min)/(max - min)

def gradient_ascent_full(learning_rate): 
  w0, w1, w2, w3 = 0,0,0,0
  for k in range(1000):
    djw0, djw1, djw2, djw3 = 0,0,0,0 
    for i in range(len(train_features)):
      total = w0 + w1*train_features[i,0] + w2*train_features[i,1] + w3*train_features[i,2]
      prob = (1/(1 + math.exp(-total)))
      djw0 += train_labels[i] - prob
      djw1 += (train_labels[i] - prob)*train_features[i,0]
      djw2 += (train_labels[i] - prob)*train_features[i,1]
      djw3 += (train_labels[i] - prob)*train_features[i,2]
    w0 += learning_rate*djw0
    w1 += learning_rate*djw1
    w2 += learning_rate*djw2
    w3 += learning_rate*djw3  
  return w0, w1, w2, w3

def gradient_ascent_mini(learning_rate):
  w0 = random.normal(loc=0, scale=0.01)
  w1 = random.normal(loc=0, scale=0.01)
  w2 = random.normal(loc=0, scale=0.01)
  w3 = random.normal(loc=0, scale=0.01)
  for k in range(1000):
    djw0, djw1, djw2, djw3 = 0,0,0,0 
    batch_index = k%8
    batch = batches[batch_index]
    for index, item in enumerate(batch):
      total = w0 + w1*item[0] + w2*item[1] + w3*item[2]
      prob = 1 - (1/(1 + math.exp(total)))
      i = batch_index*100 + index
      djw0 += train_labels[i] - prob
      djw1 += (train_labels[i] - prob)*item[0]
      djw2 += (train_labels[i] - prob)*item[1]
      djw3 += (train_labels[i] - prob)*item[2]
    w0 += learning_rate*djw0
    w1 += learning_rate*djw1
    w2 += learning_rate*djw2
    w3 += learning_rate*djw3  
  return w0, w1, w2, w3 

def gradient_ascent_stoch(learning_rate):
  w0 = random.normal(loc=0, scale=0.01)
  w1 = random.normal(loc=0, scale=0.01)
  w2 = random.normal(loc=0, scale=0.01)
  w3 = random.normal(loc=0, scale=0.01)
  dataset_length = len(train_features) 
  for k in range(1000):
    djw0, djw1, djw2, djw3 = 0,0,0,0 
    i = rnd.randint(0, len(train_features)-1)
    sample = train_features[i]
    total = w0 + w1*sample[0] + w2*sample[1] + w3*sample[2]
    prob = 1 - (1/(1 + math.exp(total)))
    djw0 += train_labels[i] - prob
    djw1 += (train_labels[i] - prob)*sample[0]
    djw2 += (train_labels[i] - prob)*sample[1]
    djw3 += (train_labels[i] - prob)*sample[2]
    w0 += learning_rate*djw0
    w1 += learning_rate*djw1
    w2 += learning_rate*djw2
    w3 += learning_rate*djw3  
  return w0, w1, w2, w3

def calculate_performance(predictions, truths, t):
  correct_guesses = 0
  tp, tn, fp, fn = 0,0,0,0
  for index, prediction in enumerate(predictions):
    if prediction == truths[index]:
      if prediction == t:
        tp += 1
      else:
        tn += 1        
      correct_guesses += 1
    else:
      if prediction == t:
        fp += 1
      else:
        fn += 1

  accuracy = correct_guesses/len(truths)
  
  if tp + fp == 0:
    precision = 0
    fdr = 0
  else:
    precision = tp/(tp+fp)
    fdr = fp/(fp+tp)

  if tp+fn == 0:
    recall = 0
  else:
    recall = tp/(tp+fn)

  if tn+fn == 0:
    npv = 0
  else:
    npv = tn/(tn+fn)

  if fp+tn == 0:
    fpr = 0
  else:
    fpr = fp/(fp+tn) 

  if precision + recall == 0:
    f1 = 0
    f2 = 0
  else:
    f1 = (2*precision * recall)/(precision + recall)
    f2 = (5 * precision * recall)/(4*precision + recall)  
  
  return tp, tn, fp, fn, accuracy, precision, recall, npv, fpr, fdr, f1, f2

df_features = pd.read_csv('question-3-features-train.csv')
df_labels = pd.read_csv('question-3-labels-train.csv')
train_features = df_features.to_numpy()
train_labels = df_labels.to_numpy()
df_features2 = pd.read_csv('question-3-features-test.csv')
df_labels2 = pd.read_csv('question-3-labels-test.csv')
test_features = df_features2.to_numpy()
test_labels = df_labels2.to_numpy()

normalize(train_features[:,0])
normalize(train_features[:,1])
normalize(train_features[:,2])
normalize(test_features[:,0])
normalize(test_features[:,1])
normalize(test_features[:,2])

batch_count = int(len(train_features)/100)
if len(train_features) % 100 > 0:
  batch_count += 1

batches = []
index = 0
for i in range(batch_count - 1):
  batches.append(train_features[index:index+100])
  index += 100
batches.append(train_features[index:])
batch_index = rnd.randint(0, batch_count-1)

print("---------------Question 3.1-------------------")
w0, w1, w2, w3 = gradient_ascent_full(0.01)
correct_guesses = 0
predictions = []
for index, item in enumerate(test_features):
  total = w0 + w1*item[0] + w2*item[1] + w3*item[2]
  if total > 0:
    predictions.append(1)  
  else:
    predictions.append(0)  
  
tp, tn, fp, fn, accuracy, precision, recall, npv, fpr, fdr, f1, f2 = calculate_performance(predictions, test_labels, 1)
tp_n, tn_n, fp_n, fn_n, accuracy_n, precision_n, recall_n, npv_n, fpr_n, fdr_n, f1_n, f2_n = calculate_performance(predictions, test_labels, 0)

print("Accuracy:", accuracy)
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn) 
print("\nMetrics For Class 0:")
print("\tPrecision:", precision_n)
print("\tRecall:", recall_n)
print("\tNegative Predictive Value:", npv_n)
print("\tFalse Positive Rate:", fpr_n)
print("\tFalse Discovery Rate:", fdr_n)
print("\tF1 Score:", f1_n)
print("\tF2 Score:", f2_n)
print("\nMetrics For Class 1:")
print("\tPrecision:", precision)
print("\tRecall:", recall)
print("\tNegative Predictive Value:", npv)
print("\tFalse Positive Rate:", fpr)
print("\tFalse Discovery Rate:", fdr)
print("\tF1 Score:", f1)
print("\tF2 Score:", f2)

print("---------------Question 3.2 Mini-Batch Gradient Ascent-------------------")
w0, w1, w2, w3 = gradient_ascent_mini(0.01)
predictions = []
for index, item in enumerate(test_features):
  total = w0 + w1*item[0] + w2*item[1] + w3*item[2]
  if total > 0:
    predictions.append(1)  
  else:
    predictions.append(0)  
  

tp, tn, fp, fn, accuracy, precision, recall, npv, fpr, fdr, f1, f2 = calculate_performance(predictions, test_labels, 1)
tp_n, tn_n, fp_n, fn_n, accuracy_n, precision_n, recall_n, npv_n, fpr_n, fdr_n, f1_n, f2_n = calculate_performance(predictions, test_labels, 0)
mic_avg_precision = (tp + tp_n)/(tp + tp_n + fp + fp_n)
mic_avg_recall = (tp + tp_n)/(tp + tp_n + fn + fn_n)
mic_avg_npv = (tn + tn_n)/(tn + tn_n + fn + fn_n)
mic_avg_fpr = (fp + fp_n)/(fp + fp_n + tn + tn_n)
mic_avg_fdr = (fp + fp_n)/(fp + fp_n + tp + tp_n)
mic_avg_f1 = (2*precision * recall + 2*precision_n * recall_n)/(precision + recall + precision_n + recall_n)
mic_avg_f2 = (5*precision*recall + 5*precision_n*recall_n)/(4*precision + recall + 4*precision_n + recall_n)
mac_avg_precision = (precision + precision_n)/2
mac_avg_recall = (recall + recall_n)/2
mac_avg_npv = (npv + npv_n)/2
mac_avg_fpr = (fpr + fpr_n)/2
mac_avg_fdr = (fdr + fdr_n)/2
mac_avg_f1 = (f1 + f1_n)/2
mac_avg_f2 = (f2 + f2_n)/2

print("Accuracy:", accuracy)
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn) 
print("Micro Averages:")
print("\tPrecision:", mic_avg_precision)
print("\tRecall:", mic_avg_recall)
print("\tNegative Predictive Value:", mic_avg_npv)
print("\tFalse Positive Rate:", mic_avg_fpr)
print("\tFalse Discovery Rate:", mic_avg_fdr)
print("\tF1 Score:", mic_avg_f1)
print("\tF2 Score:", mic_avg_f2)
print("Macro Averages:")
print("\tPrecision:", mac_avg_precision)
print("\tRecall:", mac_avg_recall)
print("\tNegative Predictive Value:", mac_avg_npv)
print("\tFalse Positive Rate:", mac_avg_fpr)
print("\tFalse Discovery Rate:", mac_avg_fdr)
print("\tF1 Score:", mac_avg_f1)
print("\tF2 Score:", mac_avg_f2)

print("---------------Question 3.2 Stochastic Gradient Ascent-------------------")
predictions = []
w0, w1, w2, w3 = gradient_ascent_stoch(0.01)
for index, item in enumerate(test_features):
  total = w0 + w1*item[0] + w2*item[1] + w3*item[2]
  if total > 0:
    predictions.append(1)  
  else:
    predictions.append(0)  

tp, tn, fp, fn, accuracy, precision, recall, npv, fpr, fdr, f1, f2 = calculate_performance(predictions, test_labels, 1)
tp_n, tn_n, fp_n, fn_n, accuracy_n, precision_n, recall_n, npv_n, fpr_n, fdr_n, f1_n, f2_n = calculate_performance(predictions, test_labels, 0)
mic_avg_precision = (tp + tp_n)/(tp + tp_n + fp + fp_n)
mic_avg_recall = (tp + tp_n)/(tp + tp_n + fn + fn_n)
mic_avg_npv = (tn + tn_n)/(tn + tn_n + fn + fn_n)
mic_avg_fpr = (fp + fp_n)/(fp + fp_n + tn + tn_n)
mic_avg_fdr = (fp + fp_n)/(fp + fp_n + tp + tp_n)
mic_avg_f1 = (2*precision * recall + 2*precision_n * recall_n)/(precision + recall + precision_n + recall_n)
mic_avg_f2 = (5*precision*recall + 5*precision_n*recall_n)/(4*precision + recall + 4*precision_n + recall_n)
mac_avg_precision = (precision + precision_n)/2
mac_avg_recall = (recall + recall_n)/2
mac_avg_npv = (npv + npv_n)/2
mac_avg_fpr = (fpr + fpr_n)/2
mac_avg_fdr = (fdr + fdr_n)/2
mac_avg_f1 = (f1 + f1_n)/2
mac_avg_f2 = (f2 + f2_n)/2

print("Accuracy:", accuracy)
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn) 
print("Micro Averages:")
print("\tPrecision:", mic_avg_precision)
print("\tRecall:", mic_avg_recall)
print("\tNegative Predictive Value:", mic_avg_npv)
print("\tFalse Positive Rate:", mic_avg_fpr)
print("\tFalse Discovery Rate:", mic_avg_fdr)
print("\tF1 Score:", mic_avg_f1)
print("\tF2 Score:", mic_avg_f2)
print("Macro Averages:")
print("\tPrecision:", mac_avg_precision)
print("\tRecall:", mac_avg_recall)
print("\tNegative Predictive Value:", mac_avg_npv)
print("\tFalse Positive Rate:", mac_avg_fpr)
print("\tFalse Discovery Rate:", mac_avg_fdr)
print("\tF1 Score:", mac_avg_f1)
print("\tF2 Score:", mac_avg_f2)