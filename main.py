import scipy.io as sio
import numpy as np
from sklearn.decomposition import NMF 
from matplotlib import pyplot as plt

def get_x():
    hs_image = sio.loadmat("data/Urban_R162.mat")
    X = np.array(hs_image["Y"]).astype(np.float64)

def get_w_gt(endmembers = 6):
    if(endmembers == 6):
        hs_gt_image = sio.loadmat("data/groundTruth_Urban_end6/end6_groundTruth.mat")
        W_gt6 = hs_gt_image["M"]
        return W_gt6
    elif(endmembers == 5):
        hs_gt_image = sio.loadmat("data/groundTruth_Urban_end5/end5_groundTruth.mat")
        W_gt5 = hs_gt_image["M"]
        return W_gt5
    else:
        hs_gt_image = sio.loadmat("data/groundTruth/end4_groundTruth.mat")
        W_gt4 = hs_gt_image["M"]
        return W_gt4
    
def plot_endmembers(W, n_comps, title="Endmembers", ordering=None ):
    if(ordering == None): ordering = range(n_comps)
    W_t = W.transpose()
    bands = range(162)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    plt.xlabel("Bands")
    plt.ylabel("Reflectance")
    plt.title(title)
    for i in range(n_comps):
        plt.plot(bands, W_t[ordering[i]], color = colors[i])
    
    pathname = title + ".png"
    plt.savefig(pathname)

