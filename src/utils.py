import numpy as np
from sklearn.utils import shuffle


# Splits X, y into labeled, unlabeled and test sets, based on seeds
#                              l - the number of labeled data
#                              u - the number of unlabled data
#                              v - the number of validation data
#                              t - the number of test data
#                     test_state - seed to use to determine test/train sets
#                    label_state - seed to use to determine labeled/unlabeled/validation sets within train set
def split_data(X, y, l, u, v, t, test_state=0, label_state=0):
    
    assert X.shape[0] >= l+u+v+t, "Not enough data to create subsets"
    
    # Separate test set from training set
    X_shuf, y_shuf = shuffle(X, y, random_state=test_state)
    X_train = X_shuf[t:t+l+u+v]
    y_train = y_shuf[t:t+l+u+v]
    X_t = X_shuf[:t]
    y_t = y_shuf[:t]
    
    # Separate labeled/unlabeled/validation sets within train set
    X_train_shuf, y_train_shuf = shuffle(X_train, y_train, random_state=label_state)
    X_l = X_train_shuf[:l]
    y_l = y_train_shuf[:l]
    X_u = X_train_shuf[l:l+u]
    y_u = y_train_shuf[l:l+u]
    X_v = X_train_shuf[l+u:l+u+v]
    y_v = y_train_shuf[l+u:l+u+v]
    
    return X_l, y_l, X_u, y_u, X_v, y_v, X_t, y_t


# Normalises data to unit gaussian
def normalise_data(X, y):

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    
    # Remove undefined features
    X_isnan = np.any(np.isnan(X), axis=0)
    X_mean, X_std, X = X_mean[~X_isnan], X_std[~X_isnan], X[:, ~X_isnan]

    y_mean = np.mean(y, axis=0)[0]
    y_std = np.std(y, axis=0)[0]
    y = (y - y_mean) / y_std    

    return X, y, X_mean, y_mean, X_std, y_std


      
# Get a dataset
def get_data(data_name):
    data_dict = {
        "chlorophyll_synthetic":{"X": "data/chlorophyll_synthetic/X.npy",       "y": "data/chlorophyll_synthetic/y.npy"},
        "chlorophyll_real":     {"X": "data/chlorophyll_real/X.npy",            "y": "data/chlorophyll_real/y.npy"},
        "1D_gaussians":         {"X": "data/gaussians_synthetic/X_1D.npy",      "y": "data/gaussians_synthetic/y_1D.npy"},
        "2D_gaussians":         {"X": "data/gaussians_synthetic/X_2D.npy",      "y": "data/gaussians_synthetic/y_2D.npy"},
        "4D_gaussians":         {"X": "data/gaussians_synthetic/X_4D.npy",      "y": "data/gaussians_synthetic/y_4D.npy"},
        "8D_gaussians":         {"X": "data/gaussians_synthetic/X_8D.npy",      "y": "data/gaussians_synthetic/y_8D.npy"},
        "16D_gaussians":        {"X": "data/gaussians_synthetic/X_16D.npy",     "y": "data/gaussians_synthetic/y_16D.npy"},
        "32D_gaussians":        {"X": "data/gaussians_synthetic/X_32D.npy",     "y": "data/gaussians_synthetic/y_32D.npy"},
        "64D_gaussians":        {"X": "data/gaussians_synthetic/X_64D.npy",     "y": "data/gaussians_synthetic/y_64D.npy"},
        "128D_gaussians":       {"X": "data/gaussians_synthetic/X_128D.npy",    "y": "data/gaussians_synthetic/y_128D.npy"},
        "256D_gaussians":       {"X": "data/gaussians_synthetic/X_256D.npy",    "y": "data/gaussians_synthetic/y_256D.npy"},
        "UCI_appliances":       {"X": "data/UCI_appliances/X.npy",              "y": "data/UCI_appliances/y.npy"},
        "UCI_air_quality":      {"X": "data/UCI_air_quality/X.npy",             "y": "data/UCI_air_quality/y.npy"},
        "UCI_song_year":        {"X": "data/UCI_song_year/X.npy",               "y": "data/UCI_song_year/y.npy"},
        "UCI_electric":         {"X": "data/UCI_electric/X.npy",                "y": "data/UCI_electric/y.npy"},
        "UCI_elevators":        {"X": "data/UCI_elevators/X.npy",               "y": "data/UCI_elevators/y.npy"},
        "UCI_parkinsons":       {"X": "data/UCI_parkinsons/X.npy",              "y": "data/UCI_parkinsons/y.npy"},
        "UCI_protein":          {"X": "data/UCI_protein/X.npy",                 "y": "data/UCI_protein/y.npy"},
    }
       
    if data_name in data_dict:
        paths = data_dict[data_name]
        X = np.load(paths["X"])
        y = np.load(paths["y"])
        
        if len(X.shape) < 2:
            X = X.reshape(-1,1)
        if len(y.shape) < 2:
            y = y.reshape(-1,1)
           
        return X, y
    else:
        assert False, "ERROR: Unknown data name: \"{}\". Data names are [{}].".format(data_name, ", ".join(key for key in data_dict))

    