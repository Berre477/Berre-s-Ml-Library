import numpy as np

def accuracy_score(y_true,y_pred):
    true_arr = np.asanyarray(y_true)
    pred_arr = np.asanyarray(y_pred)

    #cal mean of matches
    score = np.mean(true_arr == pred_arr)
    return float(score)