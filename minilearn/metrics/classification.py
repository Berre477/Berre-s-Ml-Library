import numpy as np

def accuracy_score(y_true,y_pred):
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)

    #cal mean of matches
    score = np.mean(true_arr == pred_arr)
    return float(score)


def confusion_matrix(ytrue,ypred,labels=None):
    t_arr = np.asarray(ytrue)
    p_arr = np.asarray(ypred)

    #Grab unique labels from both arrays
    if labels is None:
        labels = np.unique(np.concatenate([[t_arr,p_arr]]))

    numClasses = len(labels)
    #mapping labels 
    idx_map = {}
    for i in range(numClasses):
        idx_map[labels[i]] = i

    cm = np.zeros((numClasses,numClasses),dtype=np.int64)
    #Loops through dataset // look for a faster way
    for i in range(len(t_arr)):
        actual = t_arr[i]
        predicted = p_arr[i]
        if actual in idx_map and predicted in idx_map:
            r_idx = idx_map[actual]
            c_idx = idx_map[predicted]
            cm[r_idx,c_idx] +=1
    return cm