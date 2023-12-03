from libnmf.gnmf import GNMF


def gnmf(X, rank):
    gnmf_model = GNMF(X, rank)
    gnmf_model.compute_factors(max_iter=20, lmd=0.3, weight_type='heat-kernel', param= 0.4)
    return gnmf_model.W, gnmf_model.H, gnmf_model.div_error