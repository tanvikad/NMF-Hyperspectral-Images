from scipy import spatial
from itertools import permutations
import numpy as np 
from scipy.stats import pearsonr
from scipy.spatial.distance import hamming 
from enum import Enum
from sklearn.metrics import mean_squared_error


class Similarity(Enum):
    COSINE_DISTANCE = 0
    PEARSONS_SIMILARITY = 1
    HAMMING_DISTANCE = 2 
    RMSE = 3

def find_similarity(W, W_gt, num_components=6, similarity_type=Similarity.COSINE_DISTANCE ):
    if(W.shape != W_gt.shape): return None, None
    l = list(permutations(range(num_components)))

    best_error = 1000000000000000
    permutation = None

    dp = np.full((num_components, num_components), -1.0)

    for perm in l:
        current_error = 0
        for i in range(len(perm)):
            if(dp[perm[i]][i] == -1):
                if(similarity_type == Similarity.COSINE_DISTANCE):
                    dp[perm[i]][i] = cosine_distance(W[:,perm[i]], W_gt[:,i])
                elif(similarity_type == Similarity.HAMMING_DISTANCE):
                    dp[perm[i]][i] = hamming_distance(W[:,perm[i]], W_gt[:,i])
                elif(similarity_type == Similarity.RMSE):
                    dp[perm[i]][i] = RMSE(W[:,perm[i]], W_gt[:,i])
                else:
                    dp[perm[i]][i] = pearsons_similarity(W[:,perm[i]], W_gt[:,i])
            current_error += dp[perm[i]][i]
            
        
        if(current_error < best_error):
            best_error = current_error
            permutation = perm

    return best_error, permutation

def cosine_distance(vec1, vec2):
    return spatial.distance.cosine(vec1, vec2)

def pearsons_similarity(vec1, vec2):
    similarity = pearsonr(vec1, vec2)[0]
    similarity = max(0, similarity) #don't let it be less than 0
    distance = 1 - similarity #to get distance
    return distance

def hamming_distance(vec1, vec2):
    return hamming(vec1, vec2)

def RMSE(vec1,vec2):
    norm_vec1 = vec1 / np.linalg.norm(vec1) # can remove the normlizadation if you want
    norm_vec2 = vec2 / np.linalg.norm(vec2)
    return mean_squared_error(norm_vec1.transpose(), norm_vec2.transpose(), squared=False)