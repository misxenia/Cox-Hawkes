
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




#read original data

# read output
data_folder='data_LGCP_Hawkes/'
model_folder='model_LGCP_Hawkes/'
filename='output/simulation_comparison/'+data_folder+model_folder

with open(filename+'prediction_error_A.txt') as file:
	lines = file.readlines()
	print(lines)


