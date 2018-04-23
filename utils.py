import numpy as np
import pandas as pd

import keras
from keras.models import Model
from keras.layers import Input, Dense, Masking, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda
from keras.layers import LSTM, RepeatVector
from keras import backend as K

from sklearn import svm, preprocessing, metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from IPython.display import clear_output
from collections import defaultdict

import h5py

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec



def to_pandas_(features, labels):
    X = PCA(n_components=2).fit_transform(features)
    labels_ = set(labels)
    labels_ = dict(zip(labels_, range(len(labels_))))
    call2species = {'1':'PN', '2':'PN', '3':'PN', '4':'PN', 
                    'A':'BM-Kenya', 'A*':'BM-Kenya', 'KA':'BM-Kenya', 'KATR':'BM-Kenya', 'Nscrm':'BM-Kenya', 'PY':'BM-Kenya', 'BO':'BM-Kenya',
                    'h':'BM-Uganda', 'p':'BM-Uganda',
                    'A_titi':'TT', 'B':'TT', 'BS':'TT', 'Bw':'TT',  'C':'TT', 'H':'TT', 'x':'TT',  
                    'H':'CB', 'K':'CB', 'K+':'CB', 'W':'CB', 'W+':'CB', 'K+/W+':'CB',                 
                    'r':'CL', 's':'CL'}
    specie = [call2species[x] for x in labels]
    return pd.DataFrame({'labels':labels,'PC1': X[:,0], 'PC2': X[:,1], 'specie': specie})

def select_species(features, species):
    return features[features['specie'].isin(species)]

def remove_calls(features, calls):
    return features[~features['labels'].isin(calls)]

def translate_bm(features):
    features[features['labels'] == 'p']['labels'] = 'PY'
    features[features['labels'] == 'h']['labels'] = 'KA'
    return features

def trasform_calls(features, calls):
    return remove_calls(features, calls)
    #return remove_calls(translate_bm(features), calls)


call2species = {'1':'PN', '2':'PN', '3':'PN', '4':'PN', 
                'A':'BM-Kenya', 'A*':'BM-Kenya', 'KA':'BM-Kenya', 'KATR':'BM-Kenya', 'Nscrm':'BM-Kenya', 'PY':'BM-Kenya', 'BO':'BM-Kenya',
                'h':'BM-Uganda', 'p':'BM-Uganda',
                'A_titi':'TT', 'B':'TT', 'BS':'TT', 'Bw':'TT',  'C':'TT', 'H':'TT', 'x':'TT',  
                'H':'CB', 'K':'CB', 'K+':'CB', 'W':'CB', 'W+':'CB', 'K+/W+':'CB',                 
                'r':'CL', 's':'CL'}

def to_pandas_(features, labels):
    X = PCA(n_components=2).fit_transform(features)
    labels_ = set(labels)
    labels_ = dict(zip(labels_, range(len(labels_))))
    specie = [call2species[x] for x in labels]
    return pd.DataFrame({'labels':labels,'PC1': X[:,0], 'PC2': X[:,1], 'specie': specie})

def select_species(features, species):
    return features[features['specie'].isin(species)]

def remove_calls(features, calls):
    return features[~features['labels'].isin(calls)]

def translate_bm(features):
    features[features['labels'] == 'p']['labels'] = 'PY'
    features[features['labels'] == 'h']['labels'] = 'KA'
    return features

def trasform_calls(features, calls):
    return remove_calls(features, calls)
    #return remove_calls(translate_bm(features), calls


def to_pandas(features, labels): 
    features = to_pandas_(features, labels)
    features = select_species(features, species)
    features = trasform_calls(features, to_remove)
    return features

def select_features(features, labels, selected_calls):
    selection = pd.DataFrame(labels).isin(selected_calls).values.reshape(-1)
    return features[selection, :], labels[selection]



import scipy.spatial.distance
import pandas as pd

distance = scipy.spatial.distance.cosine

def compute_abx(features, labels):
    features_ = defaultdict(lambda: defaultdict(list))
    hashed_features = {}
    for label, feat in zip(labels, features):
            feat_ = tuple(float(x) for x in feat)
            hash_feat = hash(feat_)
            hashed_features[hash_feat] = feat
            features_[label][hash_feat].append(list(feat))
            
    # computing all distances for pairs a,a a,b and b,b
    distances = dict()
    for a, b in permutations(hashed_features.keys(), 2):
        dist = distance(hashed_features[a], hashed_features[b])
        distances[(a,b)] = dist
        distances[(b,a)] = dist

    res_ABX = list()                                                        
    for a_, b_ in product(features_, features_):
        if a_ == b_:
            continue        
        n = 0
        ABX = 0
        for b, (a, x) in product(features_[b_], combinations(features_[a_], 2)):                                   
            ABX +=  0.5 if distances[(a, x)] == distances[(b, x)] else int(distances[(a, x)] < distances[(b, x)])
            ABX +=  0.5 if distances[(x, a)] == distances[(b, a)] else int(distances[(x, a)] < distances[(b, a)])
            n+=2
            
        res_ABX.append([a_, b_, ABX, n])
    
    df = pd.DataFrame(res_ABX)
    df.columns= ['call_1', 'call_2', 'score', 'n']
    df['score'] = df['score'] / df['n'] 
    return df



def select_cb(results):
    ''' campbell '''
    # SAME
    same_calls = ['H', 'W', 'W+']
    res_same_k = results[results['call_2'].isin(same_calls) & results['call_1'].isin(same_calls)]
    res_same_k.loc[:, 'type'] = 'same'
    same_calls = ['K', 'K+']
    res_same_p = results[results['call_2'].isin(same_calls) & results['call_1'].isin(same_calls)]
    res_same_p.loc[:, 'type'] = 'same'

    # joining data
    res_same = res_same_k.append(res_same_p)
    
    # DIFFERENT
    calls1 = ['H', 'W', 'W+']
    calls2 = ['K', 'K+']
    res_diff_a = results[results['call_1'].isin(calls1) & results['call_2'].isin(calls2)]
    res_diff_b = results[results['call_1'].isin(calls2) & results['call_2'].isin(calls1)]

    res_diff_a.loc[:, "type"] = "different"
    res_diff_b.loc[:, "type"] = "different"
    res_diff = res_diff_a.append(res_diff_b)
    all_results = res_diff.append(res_same)
    
    return all_results


def select_bm(results):
    ''' blue monkeys '''
    # SAME
    same_calls = ['KA', 'KATR', 'h']
    res_same_k = results[results['call_2'].isin(same_calls) & results['call_1'].isin(same_calls)]
    res_same_k.loc[:, 'type'] = 'same'

    same_calls = ['PY', 'p']
    res_same_p = results[results['call_2'].isin(same_calls) & results['call_1'].isin(same_calls)]
    res_same_p.loc[:, 'type'] = 'same'

    # joining data
    res_same = res_same_k.append(res_same_p)
    
    # DIFFERENT
    calls1 = ['KA', 'KATR', 'h']
    calls2 = ['PY', 'p']
    res_diff_a = results[results['call_1'].isin(calls1) & results['call_2'].isin(calls2)]
    res_diff_b = results[results['call_1'].isin(calls2) & results['call_2'].isin(calls1)]

    res_diff_a.loc[:, "type"] = "different"
    res_diff_b.loc[:, "type"] = "different"
    res_diff = res_diff_a.append(res_diff_b)
    all_results = res_diff.append(res_same)
    
    return all_results


def select_pn(results):
    # SIMILAR 
    ### K/h
    same_calls = ['2', '3', '4']
    res_same_k = results[results['call_2'].isin(same_calls) & results['call_1'].isin(same_calls)]
    res_same_k.loc[:, 'type'] = 'same'

    # joining data
    res_same = res_same_k[:]

    # DIFFERENT
    calls1 = ['2', '3', '4']
    calls2 = ['1']
    res_diff_a = results[results['call_1'].isin(calls1) & results['call_2'].isin(calls2)]
    res_diff_b = results[results['call_1'].isin(calls2) & results['call_2'].isin(calls1)]

    res_diff_a.loc[:, "type"] = "different"
    res_diff_b.loc[:, "type"] = "different"
    res_diff = res_diff_a.append(res_diff_b)
    all_results = res_diff.append(res_same)
    
    return all_results


def select_tt(results):
    # SIMILAR 
    same_calls = ['B', 'BS', 'Bw']
    res_same_k = results[results['call_2'].isin(same_calls) & results['call_1'].isin(same_calls)]
    res_same_k.loc[:, 'type'] = 'same'

    # joining data
    res_same = res_same_k[:]


    # DIFFERENT
    calls1 = ['B', 'BS', 'Bw']
    calls2 = ['A_titi', 'C', 'x']
    res_diff_a = results[results['call_1'].isin(calls1) & results['call_2'].isin(calls2)]
    res_diff_b = results[results['call_1'].isin(calls2) & results['call_2'].isin(calls1)]

    res_diff_a.loc[:, "type"] = "different"
    res_diff_b.loc[:, "type"] = "different"
    res_diff = res_diff_a.append(res_diff_b)
    all_results = res_diff.append(res_same)
    
    return all_results


def compute_selected_abx(features, labels):
    
    def _abx(features, labels, idx):
        f = [x for x in features[idx,:]]
        l = labels[idx]    
        abx_ = compute_abx(f, l)
        return abx_
    
    # Blue Monkeys
    idx = (labels == 'p') | (labels == 'h') | (labels == 'KA') | (labels == 'KATR') | (labels =='PY')
    abx_bm = _abx(features, labels, idx)
    
    # Putty Nosed
    idx = (labels == '1') | (labels == '2') | (labels == '3') | (labels == '4') | (labels == '5')
    abx_pn = _abx(features, labels, idx)
    
    # Titi
    idx = (labels == 'A_titi') | (labels == 'B') | (labels == 'Bw') | (labels == 'BS') | (labels == 'C') 
    abx_tt = _abx(features, labels, idx)
    
    # Campbell
    idx = (labels == 'K') | (labels == 'K+') | (labels == 'H') | (labels == 'W') | (labels == 'W+')
    abx_cb = _abx(features, labels, idx)

    return abx_bm, abx_pn, abx_tt, abx_cb

def select_abx(abx_bm, abx_pn, abx_tt, abx_cb):
    splits_bm = select_bm(abx_bm)
    splits_pn = select_pn(abx_pn)
    splits_tt = select_tt(abx_tt)
    splits_cb = select_cb(abx_cb)

    return splits_bm, splits_pn, splits_tt, splits_cb

def plot_abx_from_splits(splits_all_bm, splits_all_pn, splits_all_tt, splits_all_cb):
    param = {'kind':'kde', 'xlim':(0, 1.0), 'ylim':(0,9), 
         'sharex':True, 'sharey':True} 

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    splits_all_bm.groupby(['type'])['score'].plot(ax=axes[0], **param)
    splits_all_pn.groupby(['type'])['score'].plot(ax=axes[1], **param)
    splits_all_tt.groupby(['type'])['score'].plot(ax=axes[2], **param)
    splits_all_cb.groupby(['type'])['score'].plot(ax=axes[3], **param)

    # X titles
    axes[0].legend(["different", "same"], loc=2)
    axes[0].set_title("Blue Monkey", fontsize=30)
    axes[1].set_title("Putty Nosed", fontsize=30)
    axes[2].set_title("Titi", fontsize=30)
    axes[3].set_title("Campbell", fontsize=30)

    # Y titles
    axes[0].set_xlabel("ABX score", fontsize=30)
    axes[0].set_ylabel("ALL", fontsize=30)

def plot_abx(features, labels):
    abx_all_mb, abx_all_pn, abx_all_tt, abx_all_cb = compute_selected_abx(features, labels)
    splits_all_bm, splits_all_pn, splits_all_tt, splits_all_cb = select_abx(abx_all_mb, abx_all_pn, abx_all_tt, abx_all_cb)
    plot_abx_from_splits(splits_all_bm, splits_all_pn, splits_all_tt, splits_all_cb)
    


def plot_labels(feat, title):
    param = {'kind':'scatter', 'x':'PC1', 'y':'PC2', 'sharex':True, 'sharey':True, 'legend':True, 's':10} 
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True, 
                           gridspec_kw = {'wspace':0.01, 'hspace':0.01},
                           subplot_kw={'xticks': [], 'yticks': [], 'aspect':'equal'})

    def do_fig(feat, title):
        color_specie = feat['specie'].unique()
        rgb_values = sns.color_palette("hls", len(color_specie))
        color_map = dict(zip(color_specie, rgb_values))
        grouped = feat.groupby('specie')
        for key, group in grouped:
            group.plot(ax=ax[0], label=key, color=group['specie'].map(color_map), **param)
            ax[0].set_ylabel('')
            ax[0].set_xlabel('')
            ax[0].legend(loc='lower center', ncol=8)

        color_labels = feat['labels'].unique()
        rgb_values = sns.color_palette("hls", len(color_labels))
        color_map = dict(zip(color_labels, rgb_values))
        grouped = feat.groupby('labels')
        for key, group in grouped:
            group.plot(ax=ax[1], label=key, color=group['labels'].map(color_map), **param)
            ax[1].set_ylabel('')
            ax[1].set_xlabel('')
            ax[1].legend(loc='lower center', ncol=8)
    do_fig(feat, title)
    plt.show()



class Plotter(keras.callbacks.Callback):
    def __init__(self, model, x, y, labels, encoder_layer, n_epochs, metric, val_metric, save_name):
        self.model = model
        self.encoder_layer = encoder_layer
        self.x = x
        self.y = y
        self.labels = labels
        self.n_epochs = n_epochs
        self.metric = metric
        self.val_metric = val_metric
        self.save_name = save_name
        
        self.n_components = 3
        self.pca = PCA(self.n_components)
        self.uniq_labels = set(labels)
        self.rgb_values = sns.color_palette("muted", len(self.uniq_labels))
        
    def on_train_begin(self, logs={}):
        self.i = count()
        self.metrics = defaultdict(list)
        self.x_data = []
        
    def on_epoch_end(self, epoch, logs={}):
        get_layer = K.function([self.model.layers[0].input], 
                               [self.model.layers[2].output])
        xy_ = get_layer([self.x])[0]
        xy = self.pca.fit_transform(xy_)

        for metric, result in logs.iteritems():
            self.metrics[metric].append(result)
        self.x_data.append(self.i.next())
        
        clear_output(wait=True)
        t_ = "....{}: {}".format(self.i, logs.get('loss'))
        x_ = self.x_data
        y_losses = self.metrics['loss']
        y_val_losses = self.metrics['val_loss']
        y_metric = self.metrics[self.metric]
        y_val_metric = self.metrics[self.val_metric]

        fig = plt.figure(figsize=(15, 10))
        #gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        #gs = gridspec.GridSpec(self.n_components+1, self.n_components)
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
        n = 0
        for x_n, y_n in combinations(range(self.n_components), 2):
            if x_n == y_n:
                continue
                
            ax = plt.subplot(gs[0, n])
            for label in self.uniq_labels:
                idx = self.labels == label
                c=[self.rgb_values[x] for x in self.y[idx]]
                ax.scatter(xy[idx, x_n], xy[idx, y_n], marker='.', s=40, c=c, label=label)
            n+=1

        ax1 = plt.subplot(gs[-1, :])
        ax1.semilogx(x_, y_losses, '-b', label='loss')
        ax1.semilogx(x_, y_val_losses, '--b', label='validation loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_xlim((0.0, self.n_epochs))
        ax1.legend()
        
        ax2 = ax1.twinx()
        ax2.semilogx(x_, y_metric, '-r', label='{}'.format(self.metric))
        ax2.semilogx(x_, y_val_metric, '--r', label='{}'.format(self.val_metric))
        ax2.set_ylabel('{}'.format(self.metric), color='r')
        ax2.tick_params('y', colors='r')
        ax2.set_xlim((0.0, self.n_epochs))
        ax2.legend()
        #ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=8)
        plt.tight_layout()
        #plt.savefig('{}_{:03d}.png'.format(self.save_name, x_[-1]))
        plt.show()






