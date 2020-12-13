import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import sklearn.cluster as clust
from scipy.spatial.distance import cdist
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pointbiserialr
import os


#### Import core python libraries
import pandas as pd
import pickle
import os
import numpy as np
from IPython.display import Math
from itertools import product

#### Import ML lib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          axis=1,
                          figsize=(5,4), 
                          fontsize=14,
                          cmap='Blues'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Note that normalization here by default occurs across axis=1, or across each row (true class). 
    (QQ: which metric does this correspond to, precision or recall?)
    """
    
    if normalize:
        if axis == 1:
            cm = cm.astype('float') / cm.sum(axis=axis)[:, np.newaxis] * 100
        elif axis == 0:
            cm = cm.astype('float') / cm.sum(axis=axis)[np.newaxis, :] * 100
        else:
            raise ValueError("axis must be 0 or 1")
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    fig = plt.figure(figsize=figsize)
    plt.grid(b=None)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    q_labels = ['TN', 'FP', 'FN', 'TP']
    quad_font_size = 14
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    q_i = 0
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        
        q = plt.text(j, i-0.2, q_labels[q_i], horizontalalignment='center', fontsize=quad_font_size)
        q_i += 1
        q.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='white'))
        
        t = plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                     verticalalignment="center",
                 color="#002781",
                    fontsize=fontsize)
        t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='white'))
        

    plt.tight_layout(h_pad=10, w_pad=0)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return None



def plot_performance(results):
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(1, 100)
    ax.set_ylim(0.73, 1)
    
    scoring = {'Accuracy' : make_scorer(accuracy_score), 'Precision': make_scorer(precision_score)}

    params = {
    'max_features': [4,5],
    'n_estimators': [5,10],
    'max_depth': [1,10]
    }

# Get the regular numpy array from the MaskedArray
    X_axis = np.array(results["param_max_depth"], dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std, sample_score_mean + sample_score_std,
            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
            alpha=1 if sample == 'test' else 0.7,
            label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score], linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,(X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()
    return None

def get_PBS_corr_from_cols(df,target_col,cont_cols,thresh = 0 ):
    res = dict()
    for col in cont_cols:
        correlation, pval = pointbiserialr(df[target_col],df[col])
        res[col] = correlation
    inter = pd.Series(res, name='corr').reset_index()
    inter['abs_corr'] = pd.DataFrame.abs(inter['corr'])
    inter = inter[inter['abs_corr'] > thresh ]
    fin_res = inter.sort_values('corr',ascending=False)
    fin_res = fin_res.drop(columns = ['abs_corr'])
    return(fin_res)


def get_VIF_experimental(df,thresh=0.1):
    dropped=True
    while dropped:
        variables = df.columns
        dropped = False
        r = [stats.linregress(df[variables].values, df.columns.get_loc(var)) for var in df.columns]
        r_value = r.r_value
        VIF = 1/(1-r_value)
        
    max_vif = max(VIF)
    if max_vif > thresh:
        maxloc = vif.index(max_vif)
        print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
        df = df.drop([X.columns.tolist()[maxloc]], axis=1)
        dropped=True
    return list(df)
    

def get_VIF_simple(df,cols):
    proj0 = np.asarray(df[cols[0]])
    proj1 = np.asarray(df[cols[1]])
    slope, intercept, r_value, p_value, std_err = stats.linregress(proj0,proj1)
    if(r_value==1):
        VIF = 1
    else:
        VIF = 1/(1-r_value)
    return(VIF)


def get_VIF(df):
    cols = df.columns
    size = cols.size
    mat = np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            #print([cols[i],cols[j]])
            mat[i,j] = get_VIF_simple(df,np.asarray([cols[i],cols[j]]))
    return mat


def remove_max_VIF_col(VIF,df,thresh):
    maxval = np.amax(VIF)
    if(maxval >= thresh):
        loc = np.where(VIF == maxval)[0][1]
        newdf = df.drop([df.columns.tolist()[loc]],axis=1)
    else:
        newdf = df
    return(newdf)


def drop_multicollinear_features(thresh,df):
    dropped = True    
    while dropped:
        dropped = False
        VIFs = get_VIF(df)
        maxval = np.amax(VIFs)
        if maxval > thresh:
            df = remove_max_VIF_col(VIFs,df,thresh)
            dropped = True
    return(df)