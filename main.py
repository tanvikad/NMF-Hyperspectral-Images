import scipy.io as sio
import numpy as np
from sklearn.decomposition import NMF 
from matplotlib import pyplot as plt

from metrics import find_similarity
from models.nmf_hs import nmf_hs


class NMF_Models:
    def __init__(self, num_endmembers=6):
        self.hs_image = sio.loadmat("data/Urban_R162.mat")
        self.X = np.array(self.hs_image["Y"]).astype(np.float64)
        self.num_endmembers = num_endmembers
        self.set_w_gt()
        
    def set_w_gt(self, endmembers = 6):
        hs_gt_image = None
        if(endmembers == 6):
            hs_gt_image = sio.loadmat("data/groundTruth_Urban_end6/end6_groundTruth.mat")
        elif(endmembers == 5):
            hs_gt_image = sio.loadmat("data/groundTruth_Urban_end5/end5_groundTruth.mat")
        else:
            hs_gt_image = sio.loadmat("data/groundTruth/end4_groundTruth.mat")
        
        self.W_gt = hs_gt_image["M"]
    
    def plot_endmembers(self,matrix, title="Endmembers", ordering=None):
        if(ordering == None): ordering = range(self.num_endmembers)
        W_t = matrix.transpose()
        bands = range(162)
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        plt.xlabel("Bands")
        plt.ylabel("Reflectance")
        plt.title(title)
        for i in range(self.num_endmembers):
            plt.plot(bands, W_t[ordering[i]], color = colors[i])
        
        pathname = "img/" + title + ".png"
        plt.savefig(pathname)

    def run_nmf_hs(self):
        W, H, error = nmf_hs(self.X, 1000.0, 1000, self.num_endmembers)
        err, best_ordering = find_similarity(W, self.W_gt)
        self.plot_endmembers(W, title="NMF_HS", ordering=best_ordering)
        print("The error is", err)

    def plot_gt(self):
        self.plot_endmembers(self.W_gt, title="Groundtruth")


models = NMF_Models()
models.plot_gt()
models.run_nmf_hs()