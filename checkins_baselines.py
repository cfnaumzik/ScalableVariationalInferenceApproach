import numpy as np
import os
import sys
import pandas as pd
from scipy.special import gammaln
#from pysal.contrib.gwr.gwr import GWR
#from pysal.contrib.glm.family import Gaussian,Poisson
from pysal.model.mgwr.gwr import GWR
from pysal.model.mgwr.sel_bw import Sel_BW
from pysal.model.mgwr.gwr import Poisson, Gaussian
from sklearn.ensemble import RandomForestRegressor
import statsmodels as sm
from statsmodels.genmod.generalized_linear_model import GLM

from statsmodels.genmod.families import Poisson

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

import pystan
from pystan import StanModel 

w_dir = "C:/Users/Christof Naumzik/Documents/Promotion/GeospatialAnalytics/code" 
#w_dir = '/home/patrickz/desktop/gitrepos/geospatialanalytics/code/' 
#w_dir = '/local/home/cnaumzik/GeoSpatialAnalytics/code'
<<<<<<< HEAD
w_dir = r'U:\Desktop\gitrepos\GeospatialAnalytics\code' 

=======
>>>>>>> e7e8b78f5d185b02e2281d4ecbc4ea2868628707
os.chdir(w_dir)

sys.path.insert(0, w_dir + "/code/kdd_modelling/TFP")
from checkins_data_pipeline import data_pipeline


def poisson(x, mu):
    return - mu + x*np.log(mu) - gammaln(x+1) 

# Table for results
columns = ['RMSE_IS', 'LIK_IS', 'RMSE_OOS', 'LIK_OOS']

results_tbl = pd.DataFrame(columns = columns)

# =============================================================================
# GWR with count features
# =============================================================================
X_train, X_test, _, _, y_train, y_test, _, _, _,_,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'count')

#Remove intercept
X_train = np.delete(X_train,5,1)
X_test = np.delete(X_test,5,1)
<<<<<<< HEAD
p = X_train.shape[1]
m = GWR(X_train[:,0:2], y_train, X_train[:,2:p], 15, family=Poisson(), fixed=False, kernel='gaussian')
results_oos = m.predict(X_test[:,0:2], X_test[:,2:p])
results_is = m.predict(X_train[:,0:2], X_train[:,2:p])
=======
>>>>>>> e7e8b78f5d185b02e2281d4ecbc4ea2868628707

bw  =  Sel_BW(y = y_train, X_loc = X_train[:,2:], coords = X_train[:,0:2],family=Gaussian(),spherical = False,kernel='gaussian').search(criterion='CV')
m = GWR(coords = X_train[:,0:2], y= y_train, X = X_train[:,2:], bw = bw, family=Gaussian(), kernel='gaussian',spherical = False,sigma2_v1=True)
results_is = m.fit()
results_is = m.predict(points = X_train[:,0:2], P = X_train[:,2:])
rmse_is = np.sqrt(np.mean((y_train-results_is.predictions)**2))
lik_is = np.sum(poisson(x = y_train, mu = np.clip(results_is.predictions,a_min=0.0001,a_max=None)))



results_oos = m.predict(points = X_test[:,0:2], P= X_test[:,2:],exog_scale = m.exog_scale, exog_resid = m.exog_resid)
rmse_oos = np.sqrt(np.mean((y_test-results_oos.predictions)**2))
lik_oos = np.sum(poisson(x = y_test, mu = np.clip(results_oos.predictions,a_min=0.0001,a_max=None)))
row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos},name='GWR (count)')
results =results.append(row)

# =============================================================================
# GWR with min dist features
# =============================================================================
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'min dist')
#Remove intercept
X_train = np.delete(X_train,5,1)
X_test = np.delete(X_test,5,1)

bw  =  Sel_BW(y = y_train, X_loc = X_train[:,2:], coords = X_train[:,0:2],family=Gaussian(),spherical = False,kernel='gaussian').search(criterion='CV')
m = GWR(coords = X_train[:,0:2], y= y_train, X = X_train[:,2:], bw = bw, family=Gaussian(), kernel='gaussian',spherical = False,sigma2_v1=True)
results_is = m.fit()
results_is = m.predict(points = X_train[:,0:2], P = X_train[:,2:])
rmse_is = np.sqrt(np.mean((y_train-results_is.predictions)**2))
lik_is = np.sum(poisson(x = y_train.reshape((-1,)), mu = np.clip(results_is.predictions,a_min=0.0001,a_max=None)))



results_oos = m.predict(points = X_test[:,0:2], P= X_test[:,2:],exog_scale = m.exog_scale, exog_resid = m.exog_resid)
rmse_oos = np.sqrt(np.mean((y_test-results_oos.predictions)**2))
lik_oos = np.sum(poisson(x = y_test, mu = np.clip(results_oos.predictions,a_min=0.0001,a_max=None)))
row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos},name='GWR (count)')
results =results.append(row)


# =============================================================================
# Linear Kriging with count features
# =============================================================================

X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'count')

m = GLM(endog=y_train.reshape((-1,)),exog=X_train[:,2:], family=Poisson(link= sm.genmod.families.links.log))
results = m.fit()


res = y_train.reshape((-1,)) - results.fittedvalues
#
kernel = ConstantKernel()*RBF(10, (1e-2, 1e2))+WhiteKernel()
gp = GaussianProcessRegressor(kernel = kernel, 
                              n_restarts_optimizer=1)
gp.fit(X_train[:,0:2], res)
#
pred_oos = m.predict(exog = X_test[:,2:], params = results.params) + gp.predict(X_test[:,0:2])
pred_is = m.predict(exog = X_train[:,2:], params = results.params) + gp.predict(X_train[:,0:2])
#
rmse_oos = np.sqrt(np.mean((pred_oos-y_test.reshape((-1,)))**2))
lik_oos = np.sum(poisson(x = y_test.reshape((-1,)), mu = np.clip(pred_oos,a_min=0.0001,a_max=None)))


rmse_is = np.sqrt(np.mean((pred_is-y_train.reshape((-1,)))**2))
lik_is = np.sum(poisson(x = y_train.reshape((-1,)), mu = np.clip(pred_is,a_min=0.0001,a_max=None)))



row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos}, name='Linear kriging (count)')
results_tbl = results_tbl.append(row)

# =============================================================================
# Linear Kriging with dist features
# =============================================================================
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'dist')

m = GLM(endog=y_train.reshape((-1,)),exog=X_train[:,2:], family=Poisson(link= sm.genmod.families.links.log))
results = m.fit()


res = y_train.reshape((-1,)) - results.fittedvalues
#
kernel = ConstantKernel()*RBF(10, (1e-2, 1e2))+WhiteKernel()
gp = GaussianProcessRegressor(kernel = kernel, 
                              n_restarts_optimizer=1)
gp.fit(X_train[:,0:2], res)
#
pred_oos = m.predict(exog = X_test[:,2:], params = results.params) + gp.predict(X_test[:,0:2])
pred_is = m.predict(exog = X_train[:,2:], params = results.params) + gp.predict(X_train[:,0:2])
#
rmse_oos = np.sqrt(np.mean((pred_oos-y_test.reshape((-1,)))**2))
lik_oos = np.sum(poisson(x = y_test.reshape((-1,)), mu = np.clip(pred_oos,a_min=0.0001,a_max=None)))


rmse_is = np.sqrt(np.mean((pred_is-y_train.reshape((-1,)))**2))
lik_is = np.sum(poisson(x = y_train.reshape((-1,)), mu = np.clip(pred_is,a_min=0.0001,a_max=None)))


row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos}, name='Linear kriging (dist)')
results_tbl = results_tbl.append(row)


# =============================================================================
# RF Kriging with both features
# =============================================================================
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'both')


m = RandomForestRegressor(n_estimators=500, min_samples_leaf=50)
m.fit(X_train[:,2:], y_train.reshape((-1,)))
m_pred = m.predict(X_train[:,2:])
res = y_train.reshape((-1,))-m_pred

kernel = ConstantKernel()*RBF(10, (1e-2, 1e2))+WhiteKernel()
gp = GaussianProcessRegressor(kernel = kernel, 
                              n_restarts_optimizer=1)
gp.fit(X_train[:,0:2], res)

pred_oos = m.predict(X_test[:,2:]) + gp.predict(X_test[:,0:2])
pred_is = m.predict(X_train[:,2:]) + gp.predict(X_train[:,0:2])

rmse_oos = np.sqrt(np.mean((pred_oos-y_test.reshape((-1,)))**2))
lik_oos = np.sum(poisson(x = y_test.reshape((-1,)), mu = np.clip(pred_oos,a_min=0.0001,a_max=None)))


rmse_is = np.sqrt(np.mean((pred_is-y_train.reshape((-1,)))**2))
	



row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos}, name='RF kriging (both)')
results_tbl = results_tbl.append(row)

# =============================================================================
# CAR with count features
# =============================================================================

#Compile stan model
car_model = StanModel(file = w_dir + '/code/kdd_modelling/Stan/poisson_sparse_CAR.stan')
#Define max_D for CAR models
max_D = 1
#Get data + fit model
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'count')

stan_data = dict(y_train = y_train.astype(np.int64).squeeze(), y_test = y_test.astype(np.int64).squeeze(), p = X_train.shape[1] - 2, N_train = X_train.shape[0], 
                 N_test = X_test.shape[0], X_train = X_train[:,2:], X_test = X_test[:,2:], loc_train = X_train[:,0:2], 
				 loc_test = X_test[:,0:2], max_D = max_D)

model_fit = car_model.optimizing(data = stan_data,iter = 20000,init="0")
pred_oos = model_fit['pred_test']
pred_is = model_fit['pred_train']
rmse_oos = np.sqrt(np.mean((pred_oos-y_test.squeeze())**2))
lik_oos = np.sum(model_fit['log_lik_test'])
rmse_is = np.sqrt(np.mean((pred_is-y_train.squeeze())**2))
lik_is = np.sum(model_fit['log_lik'])
row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos}, name='CAR model (count)')
results_tbl = results_tbl.append(row)

# =============================================================================
# CAR with dist features
# =============================================================================
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'dist')

stan_data = dict(y_train = y_train.astype(np.int64).squeeze(), y_test = y_test.astype(np.int64).squeeze(), p = X_train.shape[1] - 2, N_train = X_train.shape[0], 
                 N_test = X_test.shape[0], X_train = X_train[:,2:], X_test = X_test[:,2:], loc_train = X_train[:,0:2], 
				 loc_test = X_test[:,0:2], max_D = max_D)

model_fit = car_model.optimizing(data = stan_data,iter = 20000,init = "0")

pred_oos = model_fit['pred_test']
pred_is = model_fit['pred_train']

rmse_oos = np.sqrt(np.mean((pred_oos-y_test.squeeze())**2))
lik_oos = np.sum(model_fit['log_lik_test'])
rmse_is = np.sqrt(np.mean((pred_is-y_train.squeeze())**2))
lik_is = np.sum(model_fit['log_lik'])

row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos}, name='CAR model (dist)')
results_tbl = results_tbl.append(row)


# =============================================================================
# Print results
# =============================================================================

with open('checkins_baseline_results_tbl.tex','w') as lf:
    lf.write(results_tbl.to_latex(float_format=lambda x: '%.3f' % x))
