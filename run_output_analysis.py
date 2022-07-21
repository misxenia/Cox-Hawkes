

if __name__=='__main__':
    
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

	my_parser = argparse.ArgumentParser()
	my_parser.add_argument('--dataset_name', action='store', default='LGCP_Hawkes' ,type=str, required=False, help='simulated dataset')
	my_parser.add_argument('--simulation_number', action='store',default=0 , type=int, help='simulation series out of 100')
	my_parser.add_argument('--model_name', action='store',default='LGCP_Hawkes' , type=str, help='model name for inference')
	
    #num_chains, thinning

	args = my_parser.parse_args()
    
       
    #### choose simulated dataset to run inference on
	data_name = args.dataset_name;
	print('Data name',data_name)
	model_name = args.model_name
	print('Model name',model_name)
    ## making sure have got correct file paths
    #script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    
	#dset_names=list(f1_get.keys())# gets the enumerated numbers 1:1000
    #### choose one of the 1000 simulations from that dataset
	simulation_number = args.simulation_number
	#dset=f1_get[dset_names[simulation_number]]    
	#data_get = dset[:]
    #T=dset.attrs['T']
    
	numpyro.set_host_device_count(2)
        
	load_data=True

	## choose the number of dataset
	i=simulation_number


	#data_name='LGCP-Hawkes'
	with open('data/'+data_name+'.pkl', 'rb') as file:
		output_dict = dill.load(file)
		simulated_output_Hawkes=output_dict['simulated_output_Hawkes'+str(i)]
		simulated_output_Hawkes_train_test=output_dict['simulated_output_Hawkes_train_test'+str(i)]
		simulated_output_Hawkes_background=output_dict['simulated_output_background '+str(i)]	
		args_train=output_dict['args_train']
		args=output_dict['args']
		data_name=output_dict['data_name']
		a_0_true=args['a_0'] #simulated_output_background['a_0'];print(a_0_true)
		n_obs=simulated_output_Hawkes['G_tot_t'].size
		rate_xy_events_true=np.exp(a_0_true)*np.ones(n_obs)
		b_0_true=args['b_0']#simulated_output_background['b_0'];print(b_0_true)
		alpha=args['alpha']
		beta=args['beta']



	# read output
	if data_name=='LGCP_Hawkes' or 'LGCP-Hawkes':
		data_folder='data_LGCP_Hawkes/'
	if model_name=='LGCP_Hawkes' or 'LGCP-Hawkes':
		model_folder='model_LGCP_Hawkes/'
	
	filename='output/simulation_comparison/'+data_folder+model_folder

	with open(filename+'output'+str(simulation_number)+'.pkl', 'rb') as file:
		output = dill.load(file)
		#model=output['model']
		mcmc_samples=output['samples']
		mcmc=output['mcmc']
		args_train=output['args_train']
		print('Parameters estiamted by mcmc are ',mcmc_samples.keys())


	if model_name=='LGCP_only':
		args_train['background']='LGCP_only'
	elif model_name=='LGCP_Hawkes':
		args_train['background']='LGCP'





	n_train=simulated_output_Hawkes_train_test['G_tot_t_train'].size
	t_events_total=simulated_output_Hawkes_train_test['G_tot_t_train'][0]
	xy_events_total=np.array((simulated_output_Hawkes_train_test['G_tot_x_train'],simulated_output_Hawkes_train_test['G_tot_y_train'])).reshape(2,n_train)
	args_train['t_events']=t_events_total
	args_train['xy_events']=xy_events_total
	#when reading the data



	save_me=True

	if 'a_0' in mcmc_samples.keys():
	  fig, ax = plt.subplots(1, 2,figsize=(15,5))
	  ax[0].plot(mcmc_samples['a_0'])
	  ax[0].axhline(a_0_true+b_0_true,color='red')
	  ax[0].set_ylim(([mcmc_samples['a_0'].min()-.5,mcmc_samples['a_0'].max()+.5]))
	  ax[0].set_xlabel('a_0')
	  ax[1].hist(mcmc_samples['a_0'],bins=150,density=True)
	  ax[1].axvline(a_0_true+b_0_true,color='red')
	  ax[1].set_xlim(([mcmc_samples['a_0'].min()-.5,mcmc_samples['a_0'].max()+.5]))

	if save_me:
	  mypath='a_0.png'
	  plt.savefig(filename+mypath)
	  
	fig, ax = plt.subplots(1, 2,figsize=(15,5))
	if 'alpha' in mcmc_samples.keys():
	  ax[0].plot(mcmc_samples['alpha'])
	  ax[0].axhline(alpha,color='red')
	  ax[0].set_xlabel('alpha')
	  #ax[0].axhline(alpha,color='red')
	  ax[1].hist(mcmc_samples['alpha'],bins=150,density=True)
	  ax[1].axvline(alpha,color='red')
	if save_me:
	  mypath='alpha.png'
	  plt.savefig(filename+mypath)


	  fig, ax = plt.subplots(1, 2,figsize=(15,5))
	if 'sigmax_2' in mcmc_samples.keys():
	  ax[0].plot(mcmc_samples['sigmax_2'])
	  ax[0].axhline(args['sigmax_2'],color='red')
	  ax[1].hist(mcmc_samples['sigmax_2'],bins=150,density=True)
	  ax[1].axvline(args['sigmax_2'],color='red')
	  ax[0].set_xlabel('sigma_x_2')

	  if save_me:
	    mypath='sigma2_x.png'
	    plt.savefig(filename+mypath)


	    fig, ax = plt.subplots(2, 2,figsize=(15,5))
	if 'sigmax_2' in mcmc_samples.keys():
	  
	  ax[0,0].plot(mcmc_samples['alpha'])
	  ax[0,0].axhline(args['alpha'],color='red')
	  ax[0,0].set_ylabel('alpha')

	  ax[0,1].plot(mcmc_samples['beta'])
	  ax[0,1].axhline(args['beta'],color='red')
	  ax[0,1].set_ylabel('beta')

	  ax[1,0].plot(mcmc_samples['a_0'])
	  ax[1,0].axhline(args['a_0'],color='red')
	  ax[1,0].set_ylabel('a_0')

	  ax[1,1].plot(mcmc_samples['sigmax_2'])
	  ax[1,1].axhline(args['sigmax_2'],color='red')
	  ax[1,1].set_ylabel('sigma')

	  if save_me:
	    mypath='trace_plots.png'
	    plt.savefig(filename+mypath)

	fig, ax = plt.subplots(2, 2,figsize=(15,5))

	if 'sigmax_2' in mcmc_samples.keys():
	  ax[0,0].plot(mcmc_samples['sigmax_2'])
	  ax[0,0].axhline(args['sigmax_2'],color='red')
	  ax[0,1].hist(mcmc_samples['sigmax_2'],bins=150,density=True)
	  ax[0,1].axvline(args['sigmax_2'],color='red')
	if 'sigmay_2' in mcmc_samples.keys():
	  ax[1,0].plot(mcmc_samples['sigmay_2'])
	  ax[1,0].axhline(args['sigmay_2'],color='red')
	  ax[1,1].hist(mcmc_samples['sigmay_2'],bins=150,density=True)
	  ax[1,1].axvline(args['sigmay_2'],color='red')

	  if save_me:
	    mypath='sigma2_x_y.png'
	    plt.savefig(filename+mypath)

	fig, ax = plt.subplots(3, 1,figsize=(5,5))
	if 'alpha' in mcmc_samples.keys():

	  ax[0].plot(mcmc_samples['a_0'],label='a_0')
	  ax[0].axhline(a_0_true,color='red',label='a_0 true')
	  ax[0].set_ylabel('a_0')

	  ax[1].plot(mcmc_samples['alpha'],label='alpha')
	  ax[1].axhline(alpha,color='red',)
	  ax[1].set_ylabel('alpha')

	  ax[2].plot(mcmc_samples['beta'])
	  ax[2].axhline(beta,color='red',label='simulated value')
	  ax[2].set_ylabel('beta')
	  ax[2].set_xlabel('iterations')
	  plt.subplots_adjust(left=0.1,
	                      bottom=0.1, 
	                      right=0.9, 
	                      top=0.9, 
	                      wspace=0.4, 
	                      hspace=0.4)
	  plt.legend()
	  if save_me:
	    mypath='trace_plots_A.png'
	    plt.savefig(filename+mypath)


	rng_key, rng_key_pred= random.split(random.PRNGKey(2))


	if args_train['background']=='LGCP_only':
	  model_mcmc=spatiotemporal_LGCP_model
	elif args_train['background']=='Poisson':
	  model_mcmc=spatiotemporal_homogenous_poisson
	else:
	  model_mcmc=spatiotemporal_hawkes_model


	#model_mcmc=spatiotemporal_hawkes_model

	predictive = Predictive(model_mcmc, mcmc_samples)
	predictions = predictive(rng_key_pred, args=args_train)

	if args_train['background']!='constant':
	  f_t_pred=predictions["f_t"]
	  f_t_pred_mean=jnp.mean(f_t_pred, axis=0)
	  f_t_pred_mean.shape
	  f_t_hpdi = hpdi(f_t_pred, 0.9)
	  #f_t_hpdi.shape
	#f_t_pred_mean=jnp.mean(f_t_pred, axis=0)[0:T_train]
	#f_t_hpdi = hpdi(f_t_pred, 0.9)[0:T_train]

	##extract the last 500 samples and get the mean
	post_samples=500
	n_total=simulated_output_Hawkes['G_tot_x'].size
	a_0_post_mean=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total].mean())
	a_0_post_samples=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total])

	ft_true=simulated_output_Hawkes_background['f_t'].flatten()
	f_xy_true=simulated_output_Hawkes_background['f_xy'].flatten()
	rate_xy_true=simulated_output_Hawkes_background['rate_xy'].flatten()
	rate_t_true=simulated_output_Hawkes_background['rate_t'].flatten()

	#x_t=args['x_t']

	if args_train['background']!='constant':
	  #ft_true=np.zeros(args_train['n_t'])
	  #f_xy_true=np.zeros(args_train['n_xy']**2)
	  ft_true_train=ft_true[:50]
	  rate_t_pred=np.exp(predictions['f_t'])
	  rate_t_pred_mean=jnp.mean(rate_t_pred, axis=0)
	  rate_t_hpdi = hpdi(rate_t_pred, 0.9)

	  f_t_pred=predictions["f_t"]
	  f_t_pred_mean=jnp.mean(f_t_pred, axis=0)
	  f_t_hpdi = hpdi(f_t_pred, 0.9)

	  fig,ax=plt.subplots(1,3,figsize=(15,5))
	  ax[0].plot(args_train['x_t'], ft_true_train, label="ground truth: ft", color="orange")
	  ax[0].scatter(args_train['x_t'][args_train['indices_t']], ft_true_train[args_train['indices_t']], color="blue", label="true f at observed times")
	  ax[0].scatter(args_train['x_t'][args_train['indices_t']], f_t_pred_mean[args_train['indices_t']], color="red", label="estimated rate at observed times")
	  ax[0].set_xlabel('Interval number')
	  ax[0].plot(args_train['x_t'], f_t_pred_mean, color="green", label="mean estimated rate")
	  ax[0].fill_between(args_train['x_t'], f_t_hpdi[0], f_t_hpdi[1], alpha=0.4, color="palegoldenrod", label="90%CI rate")
	  ax[0].legend()

	  ax[1].plot(args_train['x_t'], np.exp(ft_true_train), label="ground truth: exp(f)", color="orange")
	  ax[1].scatter(args_train['x_t'][args_train['indices_t']], np.exp(ft_true_train[args_train['indices_t']]), color="blue", label="true exp(f) at observed times")
	  ax[1].scatter(args_train['x_t'][args_train['indices_t']], np.exp(f_t_pred_mean[args_train['indices_t']]), color="red", label="estimated exp(f) at observed times")
	  ax[1].set_xlabel('Interval number')
	  ax[1].plot(args_train['x_t'], np.exp(f_t_pred_mean), color="green", label="mean estimated exp(f)")
	  ax[1].fill_between(args_train['x_t'], np.exp(f_t_hpdi[0]), np.exp(f_t_hpdi[1]), alpha=0.4, color="palegoldenrod", label="90%CI rate")
	  ax[1].legend()

	  ax[2].plot(args_train['x_t'], np.exp(a_0_true+b_0_true+ft_true_train), label="ground truth: rate", color="orange")
	  ax[2].scatter(args_train['x_t'][args_train['indices_t']], np.exp(a_0_true+b_0_true+ft_true[args_train['indices_t']]), color="blue", label="true rate at observed times")
	  ax[2].scatter(args_train['x_t'][args_train['indices_t']], np.exp(f_t_pred_mean[args_train['indices_t']]+a_0_post_mean), color="red", label="estimated rate at observed times")
	  ax[2].set_xlabel('Interval number')
	  ax[2].plot(args_train['x_t'], np.exp(f_t_pred_mean+a_0_post_mean), color="green", label="mean estimated rate")
	  ax[2].fill_between(args_train['x_t'], np.exp(f_t_hpdi[0]+a_0_post_mean), np.exp(f_t_hpdi[1]+a_0_post_mean), alpha=0.4, color="palegoldenrod", label="90%CI rate")
	  ax[2].legend()


	  rate_xy_pred=np.exp(predictions['f_xy'])
	  rate_xy_pred_mean=jnp.mean(rate_xy_pred, axis=0)
	  rate_xy_hpdi = hpdi(rate_xy_pred, 0.9)

	  f_xy_pred=predictions["f_xy"]
	  f_xy_pred_mean=jnp.mean(f_xy_pred, axis=0)
	  f_xy_hpdi = hpdi(f_xy_pred, 0.9)

	  fig, ax = plt.subplots(2,2, figsize=(10, 10))
	  _min, _max = np.amin(f_xy_true), np.amax(f_xy_true)
	  im = ax[0,0].imshow(f_xy_true.reshape(args_train['n_xy'],args_train['n_xy']), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
	  ax[0,0].title.set_text('Simulated f_xy')
	  fig.colorbar(im, ax=ax[0])
	  #fig.show()
	  _min, _max = np.amin(f_xy_pred), np.amax(f_xy_pred)
	  im = ax[0,1].imshow(f_xy_pred_mean.reshape(args_train['n_xy'],args_train['n_xy']), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
	  ax[0,1].title.set_text('Estimated f_xy')

	  rate_xy_true_norm=rate_xy_true/np.sum(rate_xy_true)
	  _min, _max = np.amin(rate_xy_true_norm), np.amax(rate_xy_true_norm)
	  im = ax[1,0].imshow(rate_xy_true_norm.reshape(args_train['n_xy'],args_train['n_xy']), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
	  ax[1,0].title.set_text('Simulated normalized rate_xy')
	  fig.colorbar(im, ax=ax[1])
	  #fig.show()
	  rate_xy_pred_norm=rate_xy_pred_mean/np.sum(rate_xy_pred)
	  _min, _max = np.amin(rate_xy_pred_norm), np.amax(rate_xy_pred_norm)
	  im = ax[1,1].imshow(rate_xy_pred_norm.reshape(args_train['n_xy'],args_train['n_xy']), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
	  ax[1,1].title.set_text('Estimated normalized rate_xy')


	if args_train['background']!='constant':
	  #ft_true=np.zeros(args_train['n_t'])
	  #f_xy_true=np.zeros(args_train['n_xy']**2)
	  ft_true_train=ft_true[:50]
	  rate_t_pred=np.exp(predictions['f_t'])
	  rate_t_pred_mean=jnp.mean(rate_t_pred, axis=0)
	  rate_t_hpdi = hpdi(rate_t_pred, 0.9)

	  f_t_pred=predictions["f_t"]
	  f_t_pred_mean=jnp.mean(f_t_pred, axis=0)
	  f_t_hpdi = hpdi(f_t_pred, 0.9)

	  fig,ax=plt.subplots(1,2,figsize=(15,5))
	  ax[0].plot(args_train['x_t'], ft_true_train, label="ground truth: ft", color="orange")
	  ax[0].scatter(args_train['x_t'][args_train['indices_t']], ft_true_train[args_train['indices_t']], color="blue", label="true f at observed times")
	  ax[0].scatter(args_train['x_t'][args_train['indices_t']], f_t_pred_mean[args_train['indices_t']], color="red", label="estimated rate at observed times")
	  ax[0].set_xlabel('Interval number')
	  ax[0].plot(args_train['x_t'], f_t_pred_mean, color="green", label="mean estimated rate")
	  ax[0].fill_between(args_train['x_t'], f_t_hpdi[0], f_t_hpdi[1], alpha=0.4, color="palegoldenrod", label="90%CI rate")
	  ax[0].legend()

	  ax[1].plot(args_train['x_t'], np.exp(ft_true_train), label="ground truth: exp(f)", color="orange")
	  ax[1].scatter(args_train['x_t'][args_train['indices_t']], np.exp(ft_true_train[args_train['indices_t']]), color="blue", label="true exp(f) at observed times")
	  ax[1].scatter(args_train['x_t'][args_train['indices_t']], np.exp(f_t_pred_mean[args_train['indices_t']]), color="red", label="estimated exp(f) at observed times")
	  ax[1].set_xlabel('Interval number')
	  ax[1].plot(args_train['x_t'], np.exp(f_t_pred_mean), color="green", label="mean estimated exp(f)")
	  ax[1].fill_between(args_train['x_t'], np.exp(f_t_hpdi[0]), np.exp(f_t_hpdi[1]), alpha=0.4, color="palegoldenrod", label="90%CI rate")
	  ax[1].legend()


	post_samples=500
	a_0_post_mean=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total].mean())
	a_0_post_samples=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total])

	if args_train['background']!='LGCP_only':
	  alpha_post_mean=np.array(mcmc_samples['alpha'][n_total-post_samples:n_total].mean())
	  alpha_post_samples=np.array(mcmc_samples['alpha'][n_total-post_samples:n_total])
	  
	  beta_post_mean=np.array(mcmc_samples['beta'][n_total-post_samples:n_total].mean())
	  beta_post_samples=np.array(mcmc_samples['beta'][n_total-post_samples:n_total])

	  sigma_x_2_post_mean=np.array(mcmc_samples['sigmax_2'][n_total-post_samples:n_total].mean())
	  sigma_x_2_post_samples=np.array(mcmc_samples['sigmax_2'][n_total-post_samples:n_total])



	def normal_dist(mean,var,num_samples=args_train["z_dim_temporal"]):
	  z_temporal=numpyro.sample("z_temporal",dist.Normal(mean, var).expand([num_samples]))

	normal_predictive = Predictive(normal_dist, num_samples=1)
	normal_predictive_samples = normal_predictive(rng_key, mean=.5, var=2)

	#z_temporal = numpyro.sample("z_temporal", dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"])))
	z_temporal= normal_predictive(rng_key,jnp.zeros(args_train["z_dim_temporal"]), jnp.ones(args_train["z_dim_temporal"]))
	decoder_nn_temporal = vae_decoder_temporal(args_train["hidden_dim_temporal"], args["n_t"])  
	decoder_params = args_train["decoder_params_temporal"]
	v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal['z_temporal']))


	#plt.plot(np.mean(mcmc_samples['z_temporal'][100:],0))
	#plt.plot(z_temporal['z_temporal'][0])


	args_test={}
	n_simul=250
	args_test['n_t']=80
	args_test['x_t']=np.arange(0,args['T'],1)
	args_test['T']=80
	args_test['n_xy']=args['n_xy']
	args_test['x_xy']=args['x_xy']
	args_test['a_0']=a_0_post_mean
	args_test['n_samples']=n_simul
	args_test['n_total']=n_total

	if args_train['background']!='constant':
	  args_test['hidden_dim_temporal']=args_train["hidden_dim_temporal"]
	  args_test['z_dim_temporal']=args_train["z_dim_temporal"]
	  args_test['decoder_params_temporal']=args_train["decoder_params_temporal"]
	  args_test['z_dim_spatial']=args_train["z_dim_spatial"]
	  args_test['hidden_dim1_spatial']=args_train["hidden_dim1_spatial"]
	  args_test['hidden_dim2_spatial']=args_train["hidden_dim2_spatial"]
	  args_test['decoder_params_spatial']=args_train["decoder_params_spatial"]

	  args_test['a_0_post_mean']=a_0_post_mean


	def sample_GP(args,typ='posterior'):
	  a_0=a_0_post_mean

	  if typ=='prior':
	    z_temporal= numpyro.sample("z_temporal",dist.Normal(jnp.zeros(args["z_dim_temporal"])), jnp.ones(args["z_dim_temporal"]))
	  else:
	    z_temporal=numpyro.deterministic("z_temporal",np.mean(mcmc_samples['z_temporal'][args['n_total']-args['n_samples']:args['n_total']],0))

	  decoder_nn_temporal = vae_decoder_temporal(args["hidden_dim_temporal"], args["n_t"])  
	  decoder_params = args["decoder_params_temporal"]
	  v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
	  f_t = numpyro.deterministic("f_t", v_t[0:args["n_t"]])
	  rate_t = numpyro.deterministic("rate_t",jnp.exp(f_t+a_0))
	  Itot_t=numpyro.deterministic("Itot_t", jnp.sum(rate_t)/args["n_t"]*args["T"])
	  #Itot = numpyro.deterministic("Itot", v[len(x)])#Itot_t=jnp.trapz(mu_0*jnp.exp(f), back_t)
	  #f_t_events=f_t[args["indices_t"]]

	  # zero mean spatial gp
	  b_0=0
	  #numpyro.sample("b_0", dist.Normal(1,3))# this was 2,2
	  if typ=='prior':
	    z_spatial = numpyro.sample("z_spatial", dist.Normal(jnp.zeros(args["z_dim_spatial"]), jnp.ones(args["z_dim_spatial"])))
	  else:
	    z_spatial=numpyro.deterministic("z_spatial",np.mean(mcmc_samples['z_spatial'][args['n_total']-args['n_samples']:args['n_total']],0))

	  decoder_nn = vae_decoder_spatial(args["hidden_dim2_spatial"], args["hidden_dim1_spatial"], args["n_xy"])  
	  decoder_params = args["decoder_params_spatial"]
	  f_xy = numpyro.deterministic("f_xy", decoder_nn[1](decoder_params, z_spatial))
	  rate_xy = numpyro.deterministic("rate_xy",jnp.exp(f_xy+b_0))
	  Itot_xy=numpyro.deterministic("Itot_xy", jnp.sum(rate_xy)/args["n_xy"]**2)
	  #f_xy_events=f_xy[args["indices_xy"]]
	  Itot_txy_back=numpyro.deterministic("Itot_txy_back",Itot_t*Itot_xy)#jnp.sum(mu_xyt*args['T']/args['n_t']/args['n']**2))




	GP_predictive = Predictive(sample_GP, num_samples=n_simul)
	GP_predictive_samples = GP_predictive(rng_key, args_test, 'post')

	#plt.plot(np.exp(GP_predictive_samples['f_t'][0]+a_0_post_mean))
	#plt.hist(simulated_output_Hawkes['G_tot_t'][0])
	#plt.plot(GP_predictive_samples['f_t'][0])

	if save_me:
		mypath='GPt_post_test.png'
		plt.savefig(filename+mypath)

	f_xy_post_mean=GP_predictive_samples['f_xy'][0]
	#f_xy_post_di = hpdi(f_xy_post, 0.9)

	fig, ax = plt.subplots(1,1, figsize=(10, 5))
	_min, _max = np.amin(f_xy_post_mean), np.amax(f_xy_post_mean)
	n=25
	im = ax.imshow(f_xy_post_mean.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
	ax.title.set_text('Predictive test f_xy')
	fig.colorbar(im, ax=ax)
	if save_me:
		mypath='GPxy_post_test.png'
		plt.savefig(filename+mypath)


	if args_train['background']!='constant':
	  args_test['Itot_xy']=np.array(GP_predictive_samples['Itot_xy'])


	past_times=t_events_total
	past_locs=xy_events_total
	n_test=n_obs-n_train
	N_new=n_test


	lambda_0_post_samples=np.exp(a_0_post_samples)
	lambda_0_post_mean=lambda_0_post_samples.mean()


	args_test['x_t']=np.arange(50,80,1)


	
	post_mean=True
	f_t_pred_mean=jnp.array(np.mean(GP_predictive_samples['f_t'][:,:],0));
	f_xy_pred_mean=np.mean(GP_predictive_samples['f_xy'],0);

	x_min, x_max, y_min, y_max=0,1,0,1
	n_simul=100
	PREDICTIONS={};TRUE={}
	PREDICTIONS['T']=np.zeros((n_simul,n_test));PREDICTIONS['X']=np.zeros((n_simul,n_test));PREDICTIONS['Y']=np.zeros((n_simul,n_test))
	TRUE['T']=np.zeros((n_simul,n_test));TRUE['X']=np.zeros((n_simul,n_test));TRUE['Y']=np.zeros((n_simul,n_test))

	rng_key, rng_key_predict = random.split(random.PRNGKey(2))
	gp_predictive = Predictive(spatiotemporal_GP, num_samples=n_simul)
	GP_prior_samples = gp_predictive(rng_key_predict, args['T'], args['x_t'], args['x_xy'], gp_kernel=exp_sq_kernel, jitter=1e-5, a_0=0, b_0=0,  var_t=1, length_t=10, var_xy=1, length_xy=.25)



	for j in range(0,n_simul):
	  if j%20 ==0:
	    print('sample',j)
	    #simulate from the underlying process 

	  T_true_test, X_true_test, Y_true_test, T_pred_all_test, X_pred_all_test,Y_pred_all_test=simulate_spatiotemporal_hawkes_predictions(past_times, 
	    past_locs, N_new, args['x_min'], args['x_max'], args['y_min'], args['y_max'], np.exp(a_0_true), args['alpha'], args['beta'], args['sigmax_2'], GP_prior_samples['Itot_xy'][j],args['background_simulation'], np.array(GP_prior_samples['f_t'][j]))
	  
	  if args_train['background']=='LGCP':

	    if post_mean:
	      T_pred, X_pred, Y_pred, T_pred_all,X_pred_all,Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
	        past_locs, N_new, x_min, x_max, y_min, y_max,  lambda_0_post_mean, alpha_post_mean, beta_post_mean, sigma_x_2_post_mean, 
	         GP_predictive_samples['Itot_xy'][0], args_train['background'], f_t_pred_mean)   
	    else:
	      T_pred, X_pred, Y_pred, T_pred_all,X_pred_all,Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
	      past_locs, N_new, x_min, x_max, y_min, y_max,  lambda_0_post_samples[j], alpha_post_samples[j], beta_post_samples[j], sigma_x_2_post_samples[j], 
	       GP_predictive_samples['Itot_xy'], args_train['background'], np.array(GP_predictive_samples['f_t'][j])) 

	  elif args_train['background']=='Poisson':
	      ST_background_predictive = Predictive(spatiotemporal_homogenous_poisson, num_samples=args["batch_size"])
	      simulated_output_background = ST_background_predictive(rng_key_predict,args_new)
	      n_obs_back=simulated_output_background['N']
	      T_pred=np.sort(simulated_output_background['t_events'])[0,0:n_test]
	      X_pred=simulated_output_background['xy_events'][0,0:n_test,0]
	      Y_pred=simulated_output_background['xy_events'][0,:n_test,1]  
	  
	  elif args_train['background']=='constant':

	    if post_mean:
	      T_pred, X_pred, Y_pred, T_pred_all,X_pred_all,Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
	        past_locs, N_new, x_min, x_max, y_min, y_max, lambda_0_post_mean, alpha_post_mean, beta_post_mean, sigma_x_2_post_mean, 0,
	        args_train['background'])  
	    else:
	      T_pred, X_pred, Y_pred, T_pred_all,X_pred_all,Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
	        past_locs, N_new, x_min, x_max, y_min, y_max, lambda_0_post_samples[j], alpha_post_samples[j], beta_post_samples[j], sigma_x_2_post_samples[j],0, 
	        args_train['background'])  
	      
	  elif args_train['background']=='LGCP_only':
	    if post_mean:
	      N_0=n_test
	      ind_t_i, t_i, rate_t_i=rej_sampling_new(N_0, np.arange(args_train['T'], T_test,1), lambda_0_post_mean*np.exp(f_t_pred_mean[T_train:]), 30)
	      N_0 = t_i.shape[0]
	      ind_xy_i, xy_i, rate_xy_i=rej_sampling_new(N_0, args['x_xy'], GP_predictive_samples['rate_xy'][j,:], args['n_xy']**2)
	      ord=t_i.sort()
	      T_pred=t_i[ord].flatten()
	      X_pred=xy_i[:,0][ord].flatten()
	      Y_pred=xy_i[:,1][ord].flatten()
	      T_pred_all=np.concatenate((past_times,T_pred.flatten()))
	      X_pred_all=np.concatenate((past_locs[0],X_pred.flatten()))
	      Y_pred_all=np.concatenate((past_locs[1],Y_pred.flatten()))

	    else:
	      N_0=n_test
	      ind_t_i, t_i, rate_t_i=rej_sampling_new(N_0, np.arange(args_train['T'], T_test,1), lambda_0_post_samples[j]*np.exp(GP_predictive_samples['f_t'][j][args_train['T']:]), 30)
	      N_0 = t_i.shape[0]
	      ind_xy_i, xy_i, rate_xy_i=rej_sampling_new(N_0, args['x_xy'], GP_predictive_samples['rate_xy'][j,:], args['n_xy']**2)
	      ord=t_i.sort()
	      T_pred=t_i[ord].flatten()
	      X_pred=xy_i[:,0][ord].flatten()
	      Y_pred=xy_i[:,1][ord].flatten()
	      T_pred_all=np.concatenate((past_times,T_pred.flatten()))
	      X_pred_all=np.concatenate((past_locs[0],X_pred.flatten()))
	      Y_pred_all=np.concatenate((past_locs[1],Y_pred.flatten()))
	  
	  PREDICTIONS['T'][j]=np.round(T_pred)
	  PREDICTIONS['X'][j]=np.round(X_pred)
	  PREDICTIONS['Y'][j]=np.round(Y_pred)

	  TRUE['T'][j]=np.round(T_true_test)
	  TRUE['X'][j]=np.round(X_true_test)
	  TRUE['Y'][j]=np.round(Y_true_test)
	  
	  ErrorA=np.zeros(n_simul);ErrorB=np.zeros(n_simul)

	nums=[10,20,30]
	EA=np.zeros(len(nums));EB=EA;EC=EA
	EA_sum=0;EB_sum=0;EC_sum=0
	for i,n_stop in enumerate(nums):
		DIFF=np.zeros(n_simul)
		for j in range(n_simul):
		  T_true_test=TRUE['T'][j];T_true_test=np.round(T_true_test+np.random.normal(0,0.3))
		  X_true_test=TRUE['X'][j];X_true_test=np.round(X_true_test)
		  Y_true_test=TRUE['Y'][j];Y_true_test=np.round(Y_true_test)
		  
		  T_pred=PREDICTIONS['T'][j]
		  X_pred=PREDICTIONS['X'][j]
		  Y_pred=PREDICTIONS['Y'][j]
		  
		  
		  
		  indices=np.arange(n_stop)
		  Et=sq_diff(T_pred[indices],simulated_output_Hawkes_train_test['G_tot_t_test'][0,:n_stop])
		  Ey=sq_diff(Y_pred[indices],simulated_output_Hawkes_train_test['G_tot_y_test'][0,:n_stop])
		  Ex=sq_diff(X_pred[indices],simulated_output_Hawkes_train_test['G_tot_x_test'][0,:n_stop])
		  ErrorA[j]=np.sqrt(Ex+Ey)+np.sqrt(+Et)

		  Et=sq_diff(np.mean(TRUE['T'],0)[indices],T_pred[indices])
		  Ex=sq_diff(np.mean(TRUE['X'],0)[indices],X_pred[indices])
		  Ey=sq_diff(np.mean(TRUE['Y'],0)[indices],Y_pred[indices])
		  ErrorB[j]=np.sqrt(Ex+Ey)+np.sqrt(+Et)


		Et=sq_diff(np.mean(TRUE['T'],0)[indices],np.mean(PREDICTIONS['T'],0)[indices])
		Ex=sq_diff(np.mean(TRUE['X'],0)[indices],np.mean(PREDICTIONS['X'],0)[indices])
		Ey=sq_diff(np.mean(TRUE['Y'],0)[indices],np.mean(PREDICTIONS['Y'],0)[indices])

		#Et=sq_diff(simulated_output_Hawkes_train_test['G_tot_t_test'][0,:n_stop],np.mean(PREDICTIONS['T'],0)[indices])
		#Ex=sq_diff(simulated_output_Hawkes_train_test['G_tot_x_test'][0,:n_stop],np.mean(PREDICTIONS['X'],0)[indices])
		#Ey=sq_diff(simulated_output_Hawkes_train_test['G_tot_y_test'][0,:n_stop],np.mean(PREDICTIONS['Y'],0)[indices])


		ErrorC=np.sqrt(Ex+Ey)+np.sqrt(+Et)

		#print(np.mean(ErrorA))
		#print(np.mean(ErrorB))
		#print(np.mean(ErrorC))

		EA[i]=np.mean(ErrorA);EA_sum+=EA[i]
		EB[i]=np.mean(ErrorB);EB_sum+=EB[i]
		EC[i]=np.mean(ErrorC);EC_sum+=EC[i]

		#### save the predictions


		fig,ax=plt.subplots(1,1,figsize=(10,5))
		ax.plot(simulated_output_Hawkes_train_test['G_tot_t_test'][0,:],'blue',label='simulated')
		f_tpred_hpdi = hpdi(PREDICTIONS['T'], 0.9)
		f_tpred_mean=np.mean(PREDICTIONS['T'],0)

		ax.plot(f_tpred_mean,'red',label='post_pred_mean')
		ax.fill_between(np.arange(0,n_test), f_tpred_hpdi[0], f_tpred_hpdi[1], alpha=0.4, color="palegoldenrod", label="90%CI rate")
		plt.legend()


		#each column gives a different way to calculate the error for each of the 
		save_me=True
		#data_folder='data_LGCP_Hawkes/'
		#model_folder='model_LGCP_Hawkes/'
		#folder_name='simulation_comparison/'

	if save_me:
		#output_dict={}
		#output_dict['prediction_error_A '+str(simulation_number)]=np.mean(ErrorA)
		#output_dict['prediction_error_B '+str(simulation_number)]=np.mean(ErrorB)
		#output_dict['prediction_error_C '+str(simulation_number)]=np.mean(ErrorC)
	  
		#with open(filename+'prediction_error'+'.pkl', 'a+') as handle:
		#dill.dump(output_dict, handle)
		#EA='prediction_error_A'+str(simulation_number)+' '+str(np.mean(ErrorA)) +'\n'
		#EB='prediction_error_B '+str(simulation_number)+' '+str(np.mean(ErrorB)) +'\n'
		#EC='prediction_error_C '+str(simulation_number)+' '+str(np.mean(ErrorC)) +'\n'
		with open(filename+'prediction_error_A'+'.txt', 'a') as f:
			f.write(str(EA[:])+'\n')
			f.write('\n mean error A'+str(np.mean(EA))+'\n')
		with open(filename+'prediction_error_B'+'.txt', 'a') as f:			
			f.write(str(EB)+'\n')
			f.write('\n'+str(np.mean(EB))+'\n')
		with open(filename+'prediction_error_C'+'.txt', 'a') as f:						
			f.write(str(EC)+'\n')
			f.write('\n'+str(np.mean(EC))+'\n')


# visualize