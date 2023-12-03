import scipy.io as sio
import numpy as np
from sklearn.decomposition import NMF 
from matplotlib import pyplot as plt

from scipy.io import savemat
from scipy.io import loadmat

from metrics import find_similarity
from metrics import Similarity
from models.nmf_hs import nmf_hs
# from models.libNMF.knmf import knmf_model
from models.nmf_hs_l1_2 import nmf_hs_l1_2
from models.gnmf import gnmf


class NMF_Models:
    def __init__(self, num_endmembers=6, reload=False):
        self.hs_image = sio.loadmat("data/Urban_R162.mat")
        self.X = np.array(self.hs_image["Y"]).astype(np.float64)
        self.num_endmembers = num_endmembers
        self.set_w_gt()
        self.W_nmf_hs = None 
        self.W_nmf_hs_l1_2 = None 
        self.W_gnmf = None 
        if(not reload): self.load_matrices()


    def save_matrices(self):
        mat_dic = {}
        if(self.W_nmf_hs is not None):
            self.W_nmf_hs = self.W_nmf_hs.astype(np.float64) 
            mat_dic["W_nmf_hs"] = self.W_nmf_hs
        if(self.W_nmf_hs_l1_2 is not None):
            self.W_nmf_hs_l1_2 = self.W_nmf_hs_l1_2.astype(np.float64) 
            mat_dic["W_nmf_hs_l1_2"] = self.W_nmf_hs_l1_2
        
        if(self.W_gnmf is not None):
            self.W_gnmf = self.W_gnmf.astype(np.float64) 
            mat_dic["W_gnmf"] = self.W_gnmf

        savemat("generated_data/data.mat", mat_dic)

    def load_matrices(self):
        mat_dic = loadmat('generated_data/data.mat')
        if("W_nmf_hs" in mat_dic):
            self.W_nmf_hs = mat_dic["W_nmf_hs"]
        if("W_nmf_hs_l1_2" in mat_dic):
            self.W_nmf_hs_l1_2 = mat_dic["W_nmf_hs_l1_2"]
        if("W_gnmf" in mat_dic):
            self.W_gnmf = mat_dic["W_gnmf"]

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
        plt.clf()

    def run_nmf_hs(self):
        if(self.W_nmf_hs is None):
            self.W_nmf_hs, H, error = nmf_hs(self.X, 1000.0, 1000, self.num_endmembers)
            self.save_matrices()
        err, best_ordering = find_similarity(self.W_nmf_hs, self.W_gt, similarity_type=Similarity.COSINE_DISTANCE)
        self.plot_endmembers(self.W_nmf_hs, title="NMF_HS", ordering=best_ordering)
        print("The error is", err)
    

    def run_gmnf(self):
        if(self.W_gnmf is None):
            self.W_gnmf, H, error = nmf_hs(self.X, 1000.0, 1000, self.num_endmembers)
            self.save_matrices()
        err, best_ordering = find_similarity(self.W_gnmf, self.W_gt, similarity_type=Similarity.COSINE_DISTANCE)
        self.plot_endmembers(self.W_gnmf, title="GNMF", ordering=best_ordering)
        print("The error is", err)

    def run_nmf_l1_2(self):
        if(self.W_nmf_hs_l1_2 is None):
            self.W_nmf_hs_l1_2, H, error = nmf_hs_l1_2(self.X, 1000.0, 0.5, 1000, self.num_endmembers)
            self.save_matrices()
        err, best_ordering = find_similarity(self.W_nmf_hs_l1_2, self.W_gt, similarity_type=Similarity.COSINE_DISTANCE)
        self.plot_endmembers(self.W_nmf_hs_l1_2, title="NMF_HS_L1_2", ordering=best_ordering)
        print("The error is", err)
        

    # def run_knmf(self):
    #     knmf_model(self.X, self.num_endmembers)


    def plot_gt(self):
        self.plot_endmembers(self.W_gt, title="Groundtruth")


models = NMF_Models()
models.plot_gt()
models.run_nmf_hs()
# models.run_knmf()
models.run_nmf_l1_2()
models.run_gmnf()