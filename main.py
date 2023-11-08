import scipy.io as sio
import numpy as np
from sklearn.decomposition import NMF 

#145 145 220
def main():
    a = sio.loadmat("data/Indian_pines.mat")
    print(type(a))

    for key in a:
        print(key)
    
    matrix = a["indian_pines"]

    print(matrix.shape)

    first_layer = np.array(matrix[0])
    model = NMF(n_components=None,init='random',random_state=0)
    W = model.fit_transform(first_layer)
    H = model.components_

    print(W)
    print("\n")
    print(H)
    


main()