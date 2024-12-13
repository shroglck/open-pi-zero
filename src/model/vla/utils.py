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


if "__main__" == __name__:
    import matplotlib.pyplot as plt

    # Parameters
    alpha = 1.5
    beta = 1
    s = 0.999
    size = 1000  # Number of samples

    # Generate samples
    tau_samples = sample_from_transformed_beta(alpha, beta, s, size)
    plt.hist(tau_samples, bins=30, density=True, alpha=0.7, color="blue")
    plt.xlabel("τ")
    plt.ylabel("Density")
    plt.title(f"Samples from Beta((s-τ)/s; {alpha}, {beta}) with s={s}")
    plt.grid(True)
    plt.savefig("beta_samples.png")
