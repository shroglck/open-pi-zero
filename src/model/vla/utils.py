import numpy as np


def sample_from_transformed_beta(alpha, beta, s=0.999, size=1):
    """generate Gamma-distributed samples without using scipy, proposed by pi0 paper"""
    X = np.random.gamma(alpha, 1, size=size)
    Y = np.random.gamma(beta, 1, size=size)

    # compute Beta-distributed samples as Z = X / (X + Y)
    Z = X / (X + Y)

    # transform to τ domain
    tau_samples = s * (1 - Z)
    return tau_samples
