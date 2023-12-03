from scipy import spatial
from itertools import permutations
import numpy as np 
from sklearn.metrics import mean_squared_error



def find_similarity(W, W_gt, num_components=6):
    if(W.shape != W_gt.shape): return None, None
    l = list(permutations(range(num_components)))

    best_error = 1000000000000000
    permutation = None

    dp = np.full((num_components, num_components), -1.0)

    for perm in l:
        current_error = 0
        for i in range(len(perm)):
            if(dp[perm[i]][i] == -1):
                dp[perm[i]][i] = cosine_similarity(W[:,perm[i]], W_gt[:,i])
                # dp[perm[i]][i] = min_RMSE(W[:,perm[i]], W_gt[:,i]) 
                
            current_error += dp[perm[i]][i]
            
        
        if(current_error < best_error):
            best_error = current_error
            permutation = perm

    return best_error, permutation

def cosine_similarity(vec1, vec2):
    return spatial.distance.cosine(vec1, vec2)

def min_RMSE(vec1,vec2):
    norm_vec1 = vec1 / np.linalg.norm(vec1)
    norm_vec2 = vec2 / np.linalg.norm(vec2)
    return mean_squared_error(norm_vec1.transpose(), norm_vec2.transpose(), squared=False)