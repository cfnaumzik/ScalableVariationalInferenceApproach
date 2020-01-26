import numpy as np
import os
import sys
import pandas as pd

from pysal.contrib.gwr.gwr import GWR
from pysal.contrib.glm.family import Gaussian

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

import pystan
from pystan import StanModel 

w_dir = "C:/Users/patrickz/Desktop/GeospatialAnalytics/code" 
#w_dir = '/home/patrickz/desktop/gitrepos/geospatialanalytics/code/' 
#w_dir = '/local/home/cnaumzik/GeoSpatialAnalytics/code'
os.chdir(w_dir)

sys.path.insert(0, w_dir + "/code/kdd_modelling/TFP")
from realestate_data_pipeline import data_pipeline


def gaussian(x, mu, var):
    return -0.5 * (np.log(2 * np.pi) + np.log(var) + np.square(mu-x) / var)

# Table for results
columns = ['RMSE_IS', 'LIK_IS', 'RMSE_OOS', 'LIK_OOS']
models = ['GWR (count)', 'GWR (min dist)', 'Kriging (count)', 'Kriging (min dist)']
results = pd.DataFrame(columns = columns)

# =============================================================================
# GWR with count features
# =============================================================================
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'count')
print(X_train.shape[0])
print(X_train.shape[1])
p = X_train.shape[1]

coords_train = X_train[:,0:2]
X_train = X_train[:,2:p+2]

coords_test = X_test[:,0:2]
X_test = X_test[:,2:p+2]

model_oos = GWR(coords_train, y_train, X_train, 12, family=Gaussian(), fixed=False, kernel='gaussian')
results_oos = model_oos.predict(coords_test, X_test)
var_os = np.var(y_test-results_oos.predictions)

model_is = GWR(coords_train, y_train, X_train, 12, family=Gaussian(), fixed=False, kernel='gaussian')
results_is = model_is.predict(coords_train, X_train)
var_is = np.var(y_train-results_is.predictions)

rmse_is = np.sqrt(np.mean((y_train-results_is.predictions)**2))
lik_is = np.sum(gaussian(y_train, results_is.predictions, var_is))
rmse_oos = np.sqrt(np.mean((y_test-results_oos.predictions)**2))
lik_oos = np.sum(gaussian(y_test, results_oos.predictions, var_os))

row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos},name='GWR (count)')
results =results.append(row)

# =============================================================================
# GWR with min dist features
# =============================================================================
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'min dist')
p = X_train.shape[1]

coords_train = X_train[:,0:2]
X_train = X_train[:,2:p+2]

coords_test = X_test[:,0:2]
X_test = X_test[:,2:p+2]

model_oos = GWR(coords_train, y_train, X_train, 12, family=Gaussian(), fixed=False, kernel='gaussian')
results_oos = model_oos.predict(coords_test, X_test)
var_os = np.var(y_test-results_oos.predictions)

model_is = GWR(coords_train, y_train, X_train, 12, family=Gaussian(), fixed=False, kernel='gaussian')
results_is = model_is.predict(coords_train, X_train)
var_is = np.var(y_train-results_is.predictions)

rmse_is = np.sqrt(np.mean((y_train-results_is.predictions)**2))
lik_is = np.sum(gaussian(y_train, results_is.predictions, var_is))
rmse_oos = np.sqrt(np.mean((y_test-results_oos.predictions)**2))
lik_oos = np.sum(gaussian(y_test, results_oos.predictions, var_os))

row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos},name='GWR (min dist)')
results =results.append(row)

# =============================================================================
# Linear Kriging with count features
# =============================================================================
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'count')
coords_train = X_train[:,0:2]
X_train = X_train[:,2:]

coords_test = X_test[:,0:2]
X_test = X_test[:,2:]

y_train = y_train.squeeze()
y_test = y_test.squeeze()

m = LinearRegression(normalize=True, copy_X=True, fit_intercept=False)
m.fit(X_train, y_train)
m_pred = m.predict(X_train)
res = y_train-m_pred

kernel = ConstantKernel()*RBF(10, (1e-2, 1e2))+WhiteKernel()
gp = GaussianProcessRegressor(kernel = kernel, 
                              n_restarts_optimizer=1)
gp.fit(coords_train, res)

pred_oos = m.predict(X_test)+gp.predict(coords_test)
pred_is = m.predict(X_train)+gp.predict(coords_train)

rmse_oos = np.sqrt(np.mean((pred_oos-y_test)**2))
lik_oos = np.sum(gaussian(x = y_test, mu = pred_oos, var= rmse_oos**2))
rmse_is = np.sqrt(np.mean((pred_is-y_train)**2))
lik_is = np.sum(gaussian(x = y_train, mu = pred_is, var= rmse_is**2))


row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos}, name='Linear kriging (count)')
results = results.append(row)

# =============================================================================
# Linear Kriging with dist features
# =============================================================================
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'dist')
coords_train = X_train[:,0:2]
X_train = X_train[:,2:]

coords_test = X_test[:,0:2]
X_test = X_test[:,2:]

y_train = y_train.squeeze()
y_test = y_test.squeeze()

m = LinearRegression(normalize=True, copy_X=True, fit_intercept=False)
m.fit(X_train, y_train)
m_pred = m.predict(X_train)
res = y_train-m_pred

kernel = ConstantKernel()*RBF(10, (1e-2, 1e2))+WhiteKernel()
gp = GaussianProcessRegressor(kernel = kernel, 
                              n_restarts_optimizer=1)
gp.fit(coords_train, res)

pred_oos = m.predict(X_test)+gp.predict(coords_test)
pred_is = m.predict(X_train)+gp.predict(coords_train)

rmse_oos = np.sqrt(np.mean((pred_oos-y_test)**2))
lik_oos = np.sum(gaussian(x = y_test, mu = pred_oos, var= rmse_oos**2))
rmse_is = np.sqrt(np.mean((pred_is-y_train)**2))
lik_is = np.sum(gaussian(x = y_train, mu = pred_is, var= rmse_is**2))


row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos}, name='Linear kriging (dist)')
results = results.append(row)

# =============================================================================
# RF Kriging with both features
# =============================================================================
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'both')
coords_train = X_train[:,0:2]
X_train = X_train[:,2:]

coords_test = X_test[:,0:2]
X_test = X_test[:,2:]

y_train = y_train.squeeze()
y_test = y_test.squeeze()

m = RandomForestRegressor(n_estimators=500, min_samples_leaf=50)
m.fit(X_train, y_train)
m_pred = m.predict(X_train)
res = y_train-m_pred

kernel = ConstantKernel()*RBF(10, (1e-2, 1e2))+WhiteKernel()
gp = GaussianProcessRegressor(kernel = kernel, 
                              n_restarts_optimizer=1)
gp.fit(coords_train, res)

pred_oos = m.predict(X_test)+gp.predict(coords_test)
pred_is = m.predict(X_train)+gp.predict(coords_train)

rmse_oos = np.sqrt(np.mean((pred_oos-y_test)**2))
lik_oos = np.sum(gaussian(x = y_test, mu = pred_oos, var= rmse_oos**2))
rmse_is = np.sqrt(np.mean((pred_is-y_train)**2))
lik_is = np.sum(gaussian(x = y_train, mu = pred_is, var= rmse_is**2))


row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos}, name='RF kriging (both)')
results = results.append(row)

# =============================================================================
# CAR with count features
# =============================================================================

#Compile stan model
car_model = StanModel(file = w_dir + '/code/kdd_modelling/Stan/normal_sparse_CAR.stan')
#Define max_D for CAR models
max_D = 1
#Get data + fit model
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'count')

stan_data = dict(y_train = y_train.squeeze(), y_test = y_test.squeeze(), p = X_train.shape[1] - 2, N_train = X_train.shape[0], 
                 N_test = X_test.shape[0], X_train = X_train[:,2:], X_test = X_test[:,2:], loc_train = X_train[:,0:2], 
				 loc_test = X_test[:,0:2], max_D = max_D)

model_fit = car_model.optimizing(data = stan_data,iter = 20000,seed = 1)
pred_oos = model_fit['pred_test']
pred_is = model_fit['pred_train']
rmse_oos = np.sqrt(np.mean((pred_oos-y_test.squeeze())**2))
lik_oos = np.sum(model_fit['log_lik_test'])
rmse_is = np.sqrt(np.mean((pred_is-y_train.squeeze())**2))
lik_is = np.sum(model_fit['log_lik'])
row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos}, name='CAR model (count)')
results = results.append(row)

# =============================================================================
# CAR with dist features
# =============================================================================
X_train, X_test, _, y_train, y_test, _, _, _,_,_ = data_pipeline(feature_engineering = True, 
                                                                 feature_type = 'dist')

stan_data = dict(y_train = y_train.squeeze(), y_test = y_test.squeeze(), p = X_train.shape[1] - 2, N_train = X_train.shape[0], 
                 N_test = X_test.shape[0], X_train = X_train[:,2:], X_test = X_test[:,2:], loc_train = X_train[:,0:2], 
				 loc_test = X_test[:,0:2], max_D = max_D)

model_fit = car_model.optimizing(data = stan_data,iter = 20000,seed = 1)

pred_oos = model_fit['pred_test']
pred_is = model_fit['pred_train']

rmse_oos = np.sqrt(np.mean((pred_oos-y_test.squeeze())**2))
lik_oos = np.sum(model_fit['log_lik_test'])
rmse_is = np.sqrt(np.mean((pred_is-y_train.squeeze())**2))
lik_is = np.sum(model_fit['log_lik'])

row = pd.Series({'RMSE_IS':rmse_is,'LIK_IS':lik_is,'RMSE_OOS':rmse_oos,'LIK_OOS':lik_oos}, name='CAR model (dist)')
results = results.append(row)


# =============================================================================
# Print results
# =============================================================================

with open('gaussian_baseline_results_tbl.tex','w') as lf:
    lf.write(results.to_latex(float_format=lambda x: '%.3f' % x))