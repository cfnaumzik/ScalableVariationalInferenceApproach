def data_pipeline(feature_engineering = False, feature_type = None):
    import numpy as np
    import pandas as pd
    import scipy.spatial.distance
   
    #Loading POI data for base + GP modelling
    poi = pd.read_csv(r'data/melbourne_pois.csv')
    housing = pd.read_csv(r'data/melbourne_data.csv')

    housing = housing[housing['Type'] == 'h']

    #Remove NAs in relevant variables and outliers
    housing = housing[["Price", "Longitude", "Latitude", "bedrooms", "bathrooms", "parkingSpots", 
                 "lotSize", "finishedArea"]].dropna()    
    q = housing["Price"].quantile(0.99)
    housing = housing[housing["Price"] < q]

    #Map coordinates on plane 
    r = 6371 #earth radius
    center = -37.814 #coordinate in center of Melbourne
    
    housing["x"] = r*np.deg2rad(housing["Longitude"])*np.cos(np.deg2rad(center))
    housing["y"] = r*np.deg2rad(housing["Latitude"])
    
    #Scaling for numerical stability of covariance matrix 
    scale_x = np.mean(housing["x"])
    scale_y = np.mean(housing["y"])
    housing["x"] = housing["x"] - scale_x
    housing["y"] = housing["y"] - scale_y
    
    #Build design matrix
    X = housing[["x", "y", "bedrooms", "bathrooms", "parkingSpots", "lotSize", "finishedArea"]].values                         

    #Response variable
    y = np.log10(housing[["Price"]].values)
    y = (y-np.mean(y))/np.std(y)
    y = y.astype(np.float64)

    #Add POI data
    typeIndicator = poi[['department_store','movie_theater','police','train_station','university']].values
                         

    poi["x"] = r*np.deg2rad(poi["lon"])*np.cos(np.deg2rad(center))
    poi["y"] = r*np.deg2rad(poi["lat"])
    
    poi["x"] = poi["x"] - scale_x
    poi["y"] = poi["y"] - scale_y
       
    locs_poi = poi[["x", "y"]].values
    coords_poi = poi[["lon", "lat"]].values
    
    #Remove spatial outlier
    typeIndicator = typeIndicator[np.sqrt(locs_poi[:,0]**2+locs_poi[:,1]**2)<=20,:]
    coords_poi = coords_poi[np.sqrt(locs_poi[:,0]**2+locs_poi[:,1]**2)<=20,:]       
    locs_poi = locs_poi[np.sqrt(locs_poi[:,0]**2+locs_poi[:,1]**2)<=20,:]   
    
    typeMinDist = np.zeros(typeIndicator.shape[0])
    for typIdx in range(typeIndicator.shape[1]): 
        typeMinDist[typIdx] = np.percentile(scipy.spatial.distance.cdist(XA = X[:,0:2], XB = locs_poi[typeIndicator[:,typIdx]==1, :]), q = 0.05)

    
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
    np.random.seed(0)
    idx = np.random.rand(X.shape[0]) < 0.8
    X_train, X_test, coords_test = X[idx,:], X[~idx,:], housing[["Longitude", "Latitude"]].values[~idx]
    y_train, y_test = y[idx,:], y[~idx,:]
    
    
    
    #Knot pointf for FITC
# =============================================================================
    M = 800
    Z = X_train[np.random.choice(X_train.shape[0], M, replace = False)]  
# =============================================================================
    
# =============================================================================
#    Z = X_train
#    M = X_train.shape[0]
# =============================================================================
        
    return (X_train, X_test, Z, y_train, y_test, poi, locs_poi, coords_poi, coords_test, typeIndicator, M, typeMinDist, scale_x, scale_y)