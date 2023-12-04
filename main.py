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
from models.rnmf import robust_nmf



class NMF_Models:
    def __init__(self, num_endmembers=6, reload=False):
        self.hs_image = sio.loadmat("data/Urban_R162.mat")
        self.X = np.array(self.hs_image["Y"]).astype(np.float64)
        self.num_endmembers = num_endmembers
        self.set_w_gt()
        self.W_nmf_hs = None 
        self.W_nmf_hs_l1_2 = None 
        self.W_gnmf = None 
        self.W_rnmf = None

        self.data_pathname = "generated_data/data" + str(self.num_endmembers) + ".mat"
        self.similarity_type = Similarity.COSINE_SIMILARITY
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
        if(self.W_rnmf is not None):
            self.W_rnmf = self.W_rnmf.astype(np.float64) 
            mat_dic["W_rnmf"] = self.W_rnmf

        savemat(self.data_pathname, mat_dic)

    def load_matrices(self):
        mat_dic = loadmat(self.data_pathname)
        if("W_nmf_hs" in mat_dic):
            self.W_nmf_hs = mat_dic["W_nmf_hs"]
        if("W_nmf_hs_l1_2" in mat_dic):
            self.W_nmf_hs_l1_2 = mat_dic["W_nmf_hs_l1_2"]
        if("W_gnmf" in mat_dic):
            self.W_gnmf = mat_dic["W_gnmf"]
        if("W_rnmf" in mat_dic):
            self.W_rnmf = mat_dic["W_rnmf"]

    def set_w_gt(self, endmembers = 6):
        hs_gt_image = None
        if(self.num_endmembers == 6):
            hs_gt_image = sio.loadmat("data/groundTruth_Urban_end6/end6_groundTruth.mat")
        elif(self.num_endmembers == 5):
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
        ax = plt.gca()
        ax.set_yticks([])
        plt.title(title)
        for i in range(self.num_endmembers):
            plt.plot(bands, W_t[ordering[i]], color = colors[i])
        
        pathname = "img/" + title + "_" + str(self.num_endmembers) + ".png"
        plt.savefig(pathname)
        plt.clf()

    def run_nmf_hs(self):
        if(self.W_nmf_hs is None):
            self.W_nmf_hs, H, error = nmf_hs(self.X, 1000.0, 1000, self.num_endmembers)
            self.save_matrices()
        err, best_ordering = find_similarity(self.W_nmf_hs, self.W_gt, num_components=self.num_endmembers, similarity_type=self.similarity_type)
        self.plot_endmembers(self.W_nmf_hs, title="NMF_HS", ordering=best_ordering)
        return err 
    
    def run_rnmf(self):
        if(self.W_rnmf is None):
            self.W_rnmf, H, outlier, obj = robust_nmf(self.X,
                                            rank=self.num_endmembers,
                                            beta=1.5,
                                            init='NMF',
                                            reg_val=1,
                                            sum_to_one=0,
                                            tol=1e-7,
                                            max_iter=50)
            self.save_matrices()
        err, best_ordering = find_similarity(self.W_rnmf, self.W_gt, num_components=self.num_endmembers, similarity_type=self.similarity_type)
        self.plot_endmembers(self.W_rnmf, title="RNMF", ordering = best_ordering)
        return err 

    def run_gmnf(self):
        if(self.W_gnmf is None):
            self.W_gnmf, H, error = nmf_hs(self.X, 1000.0, 1000, self.num_endmembers)
            self.save_matrices()

        err, best_ordering = find_similarity(self.W_gnmf, self.W_gt,num_components=self.num_endmembers, similarity_type=self.similarity_type)
        self.plot_endmembers(self.W_gnmf, title="GNMF", ordering=best_ordering)
        return err

    def run_nmf_l1_2(self):
        if(self.W_nmf_hs_l1_2 is None):
            self.W_nmf_hs_l1_2, H, error = nmf_hs_l1_2(self.X, 1000.0, 0.5, 1000, self.num_endmembers)
            self.save_matrices()
        err, best_ordering = find_similarity(self.W_nmf_hs_l1_2, self.W_gt, num_components=self.num_endmembers, similarity_type=self.similarity_type)
        self.plot_endmembers(self.W_nmf_hs_l1_2, title="NMF_HS_L1_2", ordering=best_ordering)
        return err  

    def plot_gt(self):
        self.plot_endmembers(self.W_gt, title="Groundtruth")
    

    def get_stats(self):
        self.models_names = ["RNMF", "NMF_HS", "NMF_HS_L1_2", "GNMF"]
        funcs = [self.run_rnmf, self.run_nmf_hs, self.run_nmf_l1_2, self.run_gmnf]
        self.similarities = [Similarity.COSINE_SIMILARITY, Similarity.RMSE]

        errors = [[] for i in range(len(self.similarities))]
        print("Model Name,", end="")
        for similarity in self.similarities:
            print(similarity.name, end=",")
        print("")
        for i in range(len(self.models_names)):
            print(self.models_names[i], ", ", end="")
            f = funcs[i]
            for j in range(len(self.similarities)):
                similarity = self.similarities[j]
                self.similarity_type = similarity
                err = f()
                print(err, ", ", end="")
                errors[j] += [err]
            print("")
        return errors



def get_stats_per_endmember():
    model6 = NMF_Models()
    data6 = model6.get_stats()
    model5 = NMF_Models(num_endmembers=5, reload=False)
    data5 = model5.get_stats()
    model4 = NMF_Models(num_endmembers=4, reload=False)
    data4 = model4.get_stats()

    
    num_similarities = len(model6.similarities)
    model_names = model6.models_names
    for i in range(num_similarities):
        similarity = model6.similarities[i]

        plt.xlabel("Models")
        plt.ylabel(similarity.name)

        title = similarity.name + " across 4, 5, and 6 Endmembers"
        plt.title(title)

        # ax = plt.gca()
        # ax.set_xticks(model_names)
        # ax.set_xticklabels(model_names)

        plt.plot(model_names,data4[i], '--ro', label="4 endmembers")
        plt.plot(model_names,data5[i], '--bo', label="5 endmembers")
        plt.plot(model_names,data6[i], '--go', label="6 endmembers")

        plt.legend()
        pathname = "img/" + similarity.name + "_compared.png"
        plt.savefig(pathname)
        plt.clf()
        




get_stats_per_endmember()
