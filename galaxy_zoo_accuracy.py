from scipy.stats import binom
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
import numpy as np
import matplotlib.pyplot as plt

def p_bin_over_half(n_volunteers, mu):
    return 1 - binom.cdf(k=np.ceil(n_volunteers/2.), n=n_volunteers, p=mu)

def p_mu_uniform(n_mu, mu):
    return np.ones(n_mu) / n_mu  # approximage as discrete

def possible_mu(n_mu):
    return np.arange(0, n_mu)/(2*n_mu)

def false_positive_acc(n_volunteers, n_mu, visualise=False):
    mu = possible_mu(n_mu)
    bin_over_half = list(map(lambda x: p_bin_over_half(n_volunteers, x), mu))
    p_mu = p_mu_uniform(n_mu, mu)

    if visualise:
        plt.plot(possible_mu, bin_over_half)
        plt.plot(possible_mu, p_mu)
        plt.legend(['Binomial p of mistake', 'p of mu'])
        plt.tight_layout()
    return np.sum(bin_over_half * p_mu)

if __name__ == '__main__':

    n_vols = [n for n in range(4, 40)]
    p_mistake = list(map(lambda x: false_positive_acc(x, 100), n_vols))
    plt.scatter(n_vols, p_mistake)
    plt.xlabel('Volunteers')
    plt.ylabel('P of mistaken label')
    plt.savefig('figures/p_of_mistake.png')
