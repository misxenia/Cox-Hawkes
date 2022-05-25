
## read data

import sys, dill
import numpy as np
import time 
from utils import * 
from inference_functions import *
from functions import *


data_name='LGCP-Hawkes'
load_data=True

## choose the number of dataset
i=0

if load_data:
  with open('data/'+data_name+'.pkl', 'rb') as file:
    output_dict = dill.load(file)
    simulated_output_Hawkes=output_dict['simulated_output_Hawkes'+str(i)]
    simulated_output_Hawkes_train_test=output_dict['simulated_output_Hawkes_train_test'+str(i)]
    args_train=output_dict['args_train']
    args=output_dict['args']
    data_name=output_dict['data_name']
    a_0_true=args['a_0'] #simulated_output_background['a_0'];print(a_0_true)
    n_obs=simulated_output_Hawkes['G_tot_t'].size
    rate_xy_events_true=np.exp(a_0_true)*np.ones(n_obs)
    b_0_true=args['b_0']#simulated_output_background['b_0'];print(b_0_true)




args_train['background']='LGCP'


args_train["hidden_dim_temporal"]= 50
args_train["z_dim_temporal"]= 20
## I load my 1D temporal trained decoder parameters to generate GPs with hyperparameters that make sense in this domain
# Load 
#fixed lengthscale=10, var=1, T50
with open('decoders/decoder_1d_T80_fixed_ls10', 'rb') as file:
    decoder_params = pickle.load(file)
    print(len(decoder_params))

args_train["decoder_params_temporal"] = decoder_params
#args["indices"]=indices




#@title
if args_train['background']!='constant':
  # spatial VAE training
  args_train["hidden_dim1_spatial"]= 35
  args_train["hidden_dim2_spatial"]= 30
  args_train["z_dim_spatial"]=10
  n_xy=25


#@title
if args_train['background']!='constant':
  n=n_xy
  #Load 2d spatial trained decoder
  with open('./decoders/decoder_2d_n25_infer_hyperpars'.format(n), 'rb') as file:
      decoder_params = pickle.load(file)
      print(len(decoder_params))

  args_train["decoder_params_spatial"] = decoder_params
  
  #args_train["decoder_params_spatial"]=args["decoder_params_spatial"]


# MCMC inference
args_train["num_warmup"]= 500
args_train["num_samples"] = 500
args_train["num_chains"] =1
args_train["thinning"] =2

n_train=simulated_output_Hawkes_train_test['G_tot_t_train'].size

#when reading the data
t_events_total=simulated_output_Hawkes_train_test['G_tot_t_train'][0]
xy_events_total=np.array((simulated_output_Hawkes_train_test['G_tot_x_train'],simulated_output_Hawkes_train_test['G_tot_y_train'])).reshape(2,n_train)

args_train["t_events"]=t_events_total
args_train["xy_events"]=xy_events_total


if args_train['background']!='constant':
  #need to add the indices
  indices_t=find_index(t_events_total, args_train['x_t'])
  indices_xy=find_index(xy_events_total.transpose(), args_train['x_xy'])
  args_train['indices_t']=indices_t
  args_train['indices_xy']=indices_xy
  


rng_key, rng_key_predict = random.split(random.PRNGKey(2))
rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
print(args_train['background'])
if args_train['background']=='LGCP_only':
  model_mcmc=spatiotemporal_LGCP_model
elif args_train['background']=='Poisson':
  model_mcmc=spatiotemporal_homogenous_poisson
else:
  model_mcmc=spatiotemporal_hawkes_model

args_train['x_min']=args['x_min'];
args_train['x_max']=args['x_max'];
args_train['y_min']=args['y_min'];
args_train['y_max']=args['y_max'];

if args_train['background']=='Poisson':
  args_train['a_0']=None
  args_train['t_events']=t_events_total
  args_train['xy_events']=xy_events_total
  
  args_train['b_0']=0
  args_train['t_min']=0
  args_train['t_max']=50

# inference
mcmc = run_mcmc(rng_key_post, model_mcmc, args_train)
mcmc_samples=mcmc.get_samples()

save_me=True
data_folder='data_LGCP_Hawkes/'
model_folder='model_LGCP_Hawkes/'

folder_name='simulation_comparison/'

if save_me:
  filename='output/'+folder_name+data_folder+model_folder
  import dill
  output_dict = {}
  output_dict['model']=spatiotemporal_hawkes_model
  #output_dict['guide']=guide
  output_dict['samples']=mcmc.get_samples()
  output_dict['mcmc']=mcmc
  
  with open(filename+'output.pkl', 'wb') as handle:
      dill.dump(output_dict, handle)

