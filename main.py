import scipy.io as sio


#145 145 220
def main():
    a = sio.loadmat("data/Indian_pines.mat")
    print(type(a))

    for key in a:
        print(key)
    
    matrix = a["indian_pines"]

    print(matrix.shape)

main()