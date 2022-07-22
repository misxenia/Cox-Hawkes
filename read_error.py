
import argparse
import os
import sys
import dill
import numpy as np
import time 
from utils import * 
from inference_functions import *
from functions import *

from numpyro.infer import Predictive


# I want to take the average of the column inside the 
# mean_predictions_error_A_space_10
# mean_predictions_error_A_time_10
# std_predictions_error_A_space_10
# std_predictions_error_A_time_10

#for A and B and for 10,20,30


#read original data

# read output
data_folder='data_LGCP_Hawkes/'
model_folder='model_LGCP/'
filename='output/simulation_comparison/'+data_folder+model_folder
output = []

files=[
'mean_prediction_error_A_space_10','std_prediction_error_A_space_10',
'mean_prediction_error_A_space_20','std_prediction_error_A_space_20',
'mean_prediction_error_A_space_30','std_prediction_error_A_space_30',

'mean_prediction_error_A_time_10','std_prediction_error_A_time_10',
'mean_prediction_error_A_time_20','std_prediction_error_A_time_20',
'mean_prediction_error_A_time_30','std_prediction_error_A_time_30',


'mean_prediction_error_B_space_10','std_prediction_error_B_space_10',
'mean_prediction_error_B_space_20','std_prediction_error_B_space_20',
'mean_prediction_error_B_space_30','std_prediction_error_B_space_30',


'mean_prediction_error_B_time_10','std_prediction_error_B_time_10',
'mean_prediction_error_B_time_20','std_prediction_error_B_time_20',
'mean_prediction_error_B_time_30','std_prediction_error_B_time_30']


names=[
'error_A_space_10', 'error_A_space_10','error_A_space_20','error_A_space_20', 'error_A_space_30','error_A_space_30',
'error_A_time','error_A_time', 'error_A_time', 'error_A_time','error_A_time', 'error_A_time', 
'error_B_space','error_B_space', 'error_B_space','error_B_space','error_B_space', 'error_B_space',
'error_B_time','error_B_time', 'error_B_time''error_B_time','error_B_time', 'error_B_time'
]

for i,name in enumerate(files):
	with open(filename+name+'.txt') as f:
	  myarr=np.fromfile(f, dtype=float, count=2, sep=" ")
	  num=len(myarr)
	  data = np.fromfile(f, dtype=float, count=num, sep=" ").reshape((num,1))
	
	with open(filename+'results'+'.txt', 'a') as f:
	  #print(i)
	  
	  f.write(name+': ')
	  f.write(str(np.mean(data))+'\n')
	  if i%2 ==1:
	  	f.write('\n')





