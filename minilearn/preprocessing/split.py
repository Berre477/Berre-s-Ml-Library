import numpy as np



def train_test_split(X:np.array,y:np.array,test_size:float = 0.2,shuffle:bool = True,seed:int = 42):
    n = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    if test_size > 1 or test_size < 0:raise ValueError("test_size must be betzeen 0.0 and 1.0")
    n_samples = len(X)
    if shuffle:
        rng = np.ramdom.default_rng(seed)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)


    split_idx = int(n_samples*(1.0 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]

    return X_train,X_test,y_train,y_test


    
    

