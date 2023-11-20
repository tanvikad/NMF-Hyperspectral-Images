import itertools
from sklearn.metrics import mean_squared_error


def min_RMSE(W,GT):
    '''Calculates the minimum RMSE of the endmembers.

    returns min_rms, min perm
    '''

    # Get all permutations of the rows of W_reflection
    perms = list(itertools.permutations(W_reflection.transpose()))

    # Calculate RMS for each permutation
    rms_list = []
    for perm in perms:
        rms = mean_squared_error(W_gt6.transpose(), perm, squared=False)
        rms_list.append(rms)

    # Find the permutation with the lowest RMS
    min_rms = min(rms_list)
    min_perm = perms[rms_list.index(min_rms)]

    print("Minimum RMS:", min_rms)

    return min_rms, min_perm