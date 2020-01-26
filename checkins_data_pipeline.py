def data_pipeline(feature_engineering = False, feature_type = None):
    import numpy as np
    import pandas as pd
    import scipy.spatial.distance
    from sklearn import preprocessing
    import matplotlib.pyplot as plt
    
    #Parameters
    M = 1100 # Number of knots for FITC
    r = 6371
    center = 40.73061
    np.random.seed(0)
    #Loading POI data for base + GP modelling 
    poi = pd.read_csv(r'data/poi_coordinates.csv')
    restaurants = pd.read_csv(r'data/rest_covariates.csv')
    restaurants = restaurants[["checkins_count","categories", "long", "lat", "rating","tier", "age", "Year"]].dropna()
    restaurants = restaurants[restaurants.Year >= 2009]

    onehot_enc = preprocessing.OneHotEncoder(sparse = False)
    restCat = onehot_enc.fit_transform(restaurants[['categories']].values)
    intercept = np.ones((restCat.shape[0],1))
    restCat = np.append(intercept, restCat[:,2:], axis = 1)
    
    #Mapping lat/long to coordiantes for distance calc in kernel
    restaurants["x"] = r*np.deg2rad(restaurants["long"])*np.cos(np.deg2rad(center))
    restaurants["y"] = r*np.deg2rad(restaurants["lat"])
       
    scale_x = np.mean(restaurants["x"])
    scale_y = np.mean(restaurants["y"])
    
    restaurants["x"] = restaurants["x"] - scale_x
    restaurants["y"] = restaurants["y"] - scale_y

    #Build design matrix
    X = restaurants[["x", "y","Year","rating","tier"]].values

    #scaler = preprocessing.StandardScaler()
    #X[:,3:] = scaler.fit_transform(X[:,3:])
    X = np.append(X,restCat,axis=1)  
    
    #Response variable
    y = restaurants[['checkins_count']].values/restaurants[["age"]].values
    y = np.round(y, 0)
    y = y.astype(np.float64) 
 
    #Add POI data
    onehot_enc = preprocessing.OneHotEncoder(sparse = False)
    typeIndicator = onehot_enc.fit_transform(poi[['categories']].values)
    #onehot_enc.categories_ = ['Monument', 'Movie Theater', 'Museum', 'Police', 'Shopping','Stadium', 'Train Station', 'University']
    poi["x"] = r*np.deg2rad(poi["long"])*np.cos(np.deg2rad(center))
    poi["y"] = r*np.deg2rad(poi["lat"])
    
    poi["x"] = poi["x"] - scale_x
    poi["y"] = poi["y"] - scale_y
    
    coords_poi = poi[["long", "lat"]].values
         
    locs_poi = poi[["x", "y"]].values
    poi_checkins = poi[["checkins_count"]].values.squeeze()
    poi_checkins = np.log1p(poi_checkins)
    poi_checkins = (poi_checkins-poi_checkins.mean())/poi_checkins.std()

    if(feature_engineering == True):
        def min_dist(locs, locs_poi, typIdx):
            locs_poi_j = locs_poi[typeIndicator[:,typIdx]==1,:] #select pois of type j
            d = scipy.spatial.distance.cdist(locs_poi_j, locs)
            d = np.sort(d, axis = 0)
            return(d[0,:])
        
        def poi_count(locs, locs_poi, typIdx, radius):
            locs_poi_j = locs_poi[typeIndicator[:,typIdx]==1,:] #select pois of type j
            d = scipy.spatial.distance.cdist(locs_poi_j, locs)
            poi_in_radius = d < radius
            return sum(poi_in_radius, 0)
            
        for i in range(typeIndicator.shape[1]):
            if(feature_type == 'min_dist'): 
                min_d = min_dist(locs = X[:,0:2], locs_poi = locs_poi, typIdx = i).reshape(-1,1)
                X = np.append(X, min_d, axis = 1)
            elif(feature_type == 'count'):
                count = poi_count(locs = X[:,0:2], locs_poi = locs_poi, typIdx = i, radius = 2).reshape(-1,1)         
                X = np.append(X, count, axis = 1)
            elif(feature_type == 'both'): 
                min_d = min_dist(locs = X[:,0:2], locs_poi = locs_poi, typIdx = i).reshape(-1,1)
                X = np.append(X, min_d, axis = 1)
                count = poi_count(locs = X[:,0:2], locs_poi = locs_poi, typIdx = i, radius = 2).reshape(-1,1)         
                X = np.append(X, count, axis = 1)  
    
    #split training and test
    idx = np.random.rand(X.shape[0]) < 0.8
    X_train, X_test, coords_test = X[idx,:], X[~idx,:], restaurants[["long", "lat"]].values[~idx]
    y_train, y_test = y[idx,:], y[~idx,:]   

    #Knot points for FITC
    Z = X_train[np.random.choice(X_train.shape[0], M, replace = False)]
    
    return (X_train, X_test, coords_test, Z, y_train, y_test, poi, locs_poi, coords_poi, poi_checkins, typeIndicator, M, scale_x, scale_y)

