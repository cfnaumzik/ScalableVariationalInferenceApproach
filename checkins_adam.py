import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import sys
import scipy
from scipy import special

import gpflow
from gpflow.mean_functions import MeanFunction
from gpflow.kernels import Kernel
from gpflow.decors import params_as_tensors
from gpflow.params.parameter import Parameter
from gpflow import settings
from gpflow import transforms
from gpflow.training import NatGradOptimizer, AdamOptimizer
from gpflow.likelihoods import Likelihood
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf  

plt.style.use('ggplot')
pd.set_option('display.max_columns', 500)

#w_dir = r'U:\Desktop\gitrepos\GeospatialAnalytics\code' 
w_dir = '/home/patrickz/desktop/gitrepos/geospatialanalytics/code/' 
#w_dir = '/local/home/cnaumzik/GeoSpatialAnalytics/code'
os.chdir(w_dir)

sys.path.insert(0, w_dir + "/code/kdd_modelling/TFP")
from checkins_data_pipeline import data_pipeline

save_dir = 'results/checkins/poi_kernel/'+ datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# =============================================================================
# Helper function to calculate distance-weighted POI impact:
# =============================================================================
def square_dist(X, X2):
    """
    Returns ((X - X2ᵀ)/lengthscales)².
    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.
    """
    Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
    if X2 is None:
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += Xs + tf.matrix_transpose(Xs)
        return dist

    X2s = tf.reduce_sum(tf.square(X2), axis=-1, keepdims=True)
    dist = -2 * tf.matmul(X, X2, transpose_b=True)
    dist += Xs + tf.matrix_transpose(X2s)
    return dist


class SlicedLinear(MeanFunction):
    """
    y_i = A x_i
    """
    def __init__(self, A=None, p=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        A = np.ones((1, 1)) if A is None else A
        MeanFunction.__init__(self)
        self.A = Parameter(np.atleast_2d(A), dtype=settings.float_type)
        self.p = p

    @params_as_tensors
    def __call__(self, X):
        return tf.matmul(X[:,2:self.p+2], self.A)
 
class SlicedNN(MeanFunction):
    """
    y_i = A x_i
    """
    def __init__(self, p=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        MeanFunction.__init__(self)
        self.p = p
        
        hidden_units = 8
     
        bias1 = 1
        bias2 = 1

        weights1 = np.zeros((p,hidden_units)) 
        weights2 = np.zeros((hidden_units,1)) 
        
        self.weights1   = Parameter(weights1, dtype = settings.float_type)
        self.weights2   = Parameter(weights2, dtype = settings.float_type)

        self.bias1 = Parameter(bias1, dtype = settings.float_type)
        self.bias2 = Parameter(bias2, dtype = settings.float_type)
        
    @params_as_tensors
    def __call__(self, X):
        layer1 = tf.nn.relu(tf.matmul(X[:,2:self.p+2], self.weights1) +self.bias1)
        output = tf.matmul(layer1, self.weights2) + self.bias2
        return output      
    
class POI(Kernel):
    def __init__(self, input_dim, locs_poi, typeIndicator,  
                 active_dims=None, name=None, typIdx = 1, lengthscale = None, 
                 effects = None, locs = None, mindist = 0.5, kernel_type = "linear"):
        super().__init__(input_dim, active_dims, name=name)
        effects = np.random.uniform(low = 0.5, high = 1.0) if effects is None else effects
        MeanFunction.__init__(self)
        self.locs_poi = locs_poi.astype(np.float64)
        self.typeIndicator = typeIndicator.astype(np.float64)
        self.typIdx = typIdx
        self.locs_poi_j = self.locs_poi[self.typeIndicator[:,self.typIdx]==1,:]
        self.kernel_type = kernel_type

        self.lengthscale = 0 #Parameter(2, transform=transforms.Logistic(a=mindist, b=10), dtype=settings.float_type)
        self.effects = Parameter(effects, transform=transforms.Logistic(a=0.01, b=1), dtype=settings.float_type)
        self.distmax = Parameter(0.5, transform=transforms.Logistic(a=0, b=1.5), dtype = settings.float_type) 
        
    @params_as_tensors       
    def transformed_sorted_distances(self, locs, locs2=None):
        """
        Returns h(-d(s_i, w_j)), where d is the squared distance between 
        obs and POI. If d(s_i, w_j) > d(s_i, w_{N_cutoff}), then it returns 0.
        Currently, the decay fct is h = exp(-x^2/x)
        """
        if locs2 is None:  
            d = tf.sqrt(square_dist(self.locs_poi_j, locs))
            if self.kernel_type == "Gaussian":
                d_new = self.effects*tf.exp(-0.5*(d/self.distmax)**2)
            else:
                d_new = self.effects * tf.nn.relu((-1/self.distmax)*d+1)
            out = tf.matmul(d_new, d_new, transpose_a = True)
            return(out + tf.eye(tf.shape(out)[0], dtype = tf.float64)*1e-4)
        else:
            d = tf.sqrt(square_dist(self.locs_poi_j, locs))
            d2 = tf.sqrt(square_dist(self.locs_poi_j, locs2))
            if self.kernel_type == "Gaussian":
                d_new = self.effects*tf.exp(-0.5*(d/self.distmax)**2)
                d2_new = self.effects*tf.exp(-0.5*(d2/self.distmax)**2)
            else:
                d_new = self.effects * tf.nn.relu((-1/self.distmax)*d+1)         
                d2_new = self.effects * tf.nn.relu((-1/self.distmax)*d2+1)
            out = tf.matmul(d_new, d2_new, transpose_a = True)
            return(out)
            
    @params_as_tensors
    def K(self, X, X2=None):            
        if X2 is None:
            out = self.transformed_sorted_distances(X[:,0:2])
            return out
        else:
            out = self.transformed_sorted_distances(X[:,0:2], X2[:,0:2])
            return out
    
    @params_as_tensors
    def Kdiag(self, X):
        return tf.diag_part(self.K(X))

# =============================================================================
# Build GP model 
# =============================================================================
def lpmf(a, m, x):
    Z=tf.exp(tf.lgamma(a+x)-tf.lgamma(a)-tf.lgamma(x+1.))
    return tf.log(Z)+x*tf.log(m/(m+a))+a*tf.log(a/(m+a))
  

class NegBin(Likelihood):
    def __init__(self, invlink=tf.exp, scale=1., **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink
        self.scale = Parameter(scale,transform=transforms.positive, dtype=settings.float_type)
        #self.scale = gpflow.Param(scale, transform = gpflow.transforms.positive)

    def logp(self, F, Y):
        return lpmf(x=Y, m=self.invlink(F), a=self.scale)

    @params_as_tensors
    def conditional_variance(self, F):
        return self.invlink(F)*(1+self.invlink(F)/self.scale)
    
    def conditional_mean(self, F):
        return self.invlink(F)


def init():   
    feature = mf.SharedIndependentMof(gpflow.features.InducingPoints(Z.copy()))
    #Define POI kernels
#    type_names =  ['Monument', 'Movie Theater', 'Museum', 'Police', 'Shopping','Stadium', 'Train Station', 'University']
    poi_list = [0,1,2,3,4,5,6,7]
    kern_list =[]
    for i in range(len(poi_list)):
        kern = POI(effects = 0.5, lengthscale = 5, input_dim = D, locs_poi=locs_poi,
                    typeIndicator=typeIndicator, typIdx = poi_list[i], locs = Z[:,0:2],
                    name = str(poi_list[i]),
                    kernel_type = "Linear")
        kern_list.append(kern)
       
    #Add spatial kernel
    kern_spatial = gpflow.kernels.Matern32(input_dim = D, lengthscales = 20)

    #Define kernel list    
    kern_list.append(kern_spatial)    
    L = len(kern_list)   
    W = np.ones((L,1))
    W_t = np.transpose(W)    
    kernel = mk.SeparateMixedMok(kern_list, W=W_t)    
    
    #Define linear mean function
    mean_fct = SlicedLinear(A = theta, p=p)
    #mean_fct = SlicedNN(p=p)


    q_mu = np.random.normal(0.0,1,(M, L))
    q_sqrt = np.repeat(np.eye(M)[None, ...], L, axis=0) * 1.0
    
    m = gpflow.models.SVGP(X=X_train, Y=y_train, kern=kernel, 
                           likelihood = gpflow.likelihoods.Poisson(), 
                           feat=feature, whiten = True, #minibatch_size=len(X_train),
                           mean_function = mean_fct, # + mean_poi,
                           q_mu = q_mu,
                           q_sqrt = q_sqrt,
                           name = 'svgp')
    
    #m.likelihood.scale = 1  #Initialze params
    m.feature.trainable = False
    m.kern.W.trainable = False
    return m
# =============================================================================
# Build GP model 
# =============================================================================
def test_train(iterations):   
    with gpflow.defer_build():
      m = init()
    #gamma_start = 1e-6
    #gamma_max = 1e-1
    #gamma_step = 1e-3
    #gamma = tf.Variable(gamma_start, dtype=tf.float64)
    #gamma_incremented = tf.where(tf.less(gamma, gamma_max), gamma + gamma_step, gamma_max)
    tf.local_variables_initializer()
    tf.global_variables_initializer()
    
    tf_session = m.enquire_session()
    m.compile( tf_session )
    
    bool_2600 = False
    bool_2500 = False
    bool_2450 = False

    #op_ng = NatGradOptimizer(gamma).make_optimize_tensor(m, var_list=[[m.q_mu, m.q_sqrt]])
    op_adam = AdamOptimizer(0.01).make_optimize_tensor(m)
    #op_increment_gamma = tf.assign(gamma, gamma_incremented)
    #tf_session.run(tf.variables_initializer([gamma]))
    
    for it in range(iterations):          
        try:
            tf_session.run(op_adam) 
        except tf.errors.InvalidArgumentError:
            print('Fail!')
            saver = tf.train.Saver()
            save_path = saver.save(tf_session, save_dir + "/model_fail.ckpt")
            print("Model saved in path: %s" % save_path)
            break  # we'll just stop if it fails
                          
        if it % 100 == 0:
            likelihood = tf_session.run(m.likelihood_tensor)
            print('{}, ELBO={:.4f}'.format(it, likelihood)) 
            if likelihood >= -2600: 
                if bool_2600 == False:
                    saver = tf.train.Saver()
                    save_path = saver.save(tf_session, save_dir + '/model' + '2800' +'.ckpt')
                    bool_2600 = True
                    print("Model saved in path: %s" % save_path)
                if likelihood >= -2500: 
                    if bool_2500 == False:
                        saver = tf.train.Saver()
                        save_path = saver.save(tf_session, save_dir + '/model' + '2600' +'.ckpt')
                        bool_2500 = True
                        print("Model saved in path: %s" % save_path)
                    if likelihood >= -2450 and bool_2450 == False: 
                        saver = tf.train.Saver()
                        save_path = saver.save(tf_session, save_dir + '/model' + '2500' +'.ckpt')
                        bool_2450= True
                        print("Model saved in path: %s" % save_path)       
                
    saver = tf.train.Saver()
    save_path = saver.save(tf_session, save_dir + "/model.ckpt")
    print("Model saved in path: %s" % save_path)
    
    m.anchor(tf_session)
    
    return m

# =============================================================================
# Build GP model 
# =============================================================================
def restore():   
    with gpflow.defer_build():
        m = init()
    tf.local_variables_initializer()
    tf.global_variables_initializer()
    
    tf_graph = m.enquire_graph()
    tf_session = m.enquire_session()
    m.compile( tf_session )
    
    saver = tf.train.Saver()
    save_path = saver.restore(tf_session, save_dir + "/model.ckpt")
    print("Model loaded from path: %s" % save_path)
    
    return m

# =============================================================================
# Run
# =============================================================================
X_train, X_test, coords_test, Z, y_train, y_test, poi, locs_poi, coords_poi, poi_checkins, typeIndicator, M, scale_x, scale_y = data_pipeline() # Get data

p = X_train.shape[1] - 2 #only house covariates, no POI features
D = 2 #Input dimensions for kernel
theta = np.zeros((p,1))

m = test_train(iterations = 2250)

#m = restore()

# =============================================================================
# Parameter
# =============================================================================
out_kern = m.kern.as_pandas_table()
out_kern.to_csv(save_dir + '/kern.csv', sep =';')

#out_likelihood = m.likelihood.as_pandas_table()
#out_likelihood.to_csv(save_dir + '/likelihood.csv', sep =';')

theta = m.mean_function.as_pandas_table()
theta.to_csv(save_dir +'/mean_theta.csv', sep = ';')

# =============================================================================
# Model fit
# =============================================================================

#prediction: 
mu_oos, var_oos = m.predict_y(X_test)
mu_is, var_is = m.predict_y(X_train)

rmse_oos = np.sqrt(np.mean((mu_oos-y_test)**2))
rmse_is = np.sqrt(np.mean((mu_is-y_train)**2))

print(rmse_oos)

with open(save_dir + '/out.txt', 'w') as f:
    print("RMSE OOS: ", (rmse_oos), file = f)
    print("RMSE IS: ", (rmse_is), file = f)
#    print("Total Likelihood OOS: ", sum(lpmf(x = y_test, m = mu_oos,a = m.likelihood.scale.value )), '\n', file = f)
#    print("Mean Likelihood OOS: ", np.mean(lpmf(x = y_test, m = mu_oos,a=m.likelihood.scale.value)), '\n', file = f)
#    print("Total Likelihood IS: ", sum(lpmf(x = y_train, m = mu_is,a=m.likelihood.scale.value)), '\n', file = f)
#    print("Mean Likelihood IS: ", np.mean(lpmf(x = y_train, m = mu_is,a=m.likelihood.scale.value)), '\n', file = f)
