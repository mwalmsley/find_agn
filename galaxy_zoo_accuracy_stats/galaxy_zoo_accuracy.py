from scipy.stats import binom
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
import numpy as np
import matplotlib.pyplot as plt

def p_bin_over_half(n_volunteers, mu):
    return 1 - binom.cdf(k=np.ceil(n_volunteers/2.) - 1, n=n_volunteers, p=mu)


def p_bin_under_half(n_volunteers, mu):
    return binom.cdf(k=np.ceil(n_volunteers/2.) - 1, n=n_volunteers, p=mu)


def p_mu_uniform(n_mu):
    return np.ones(n_mu) / n_mu  # approximage as discrete


def possible_mu(n_mu):
    return np.arange(0, n_mu)/(n_mu)


def false_positive_acc(n_volunteers, n_mu, visualise=False):
    mu = possible_mu(n_mu)
    bin_over_half = list(map(lambda x: p_bin_over_half(n_volunteers, x), mu))
    p_mu = p_mu_uniform(n_mu)

    if visualise:
        plt.plot(possible_mu, bin_over_half)
        plt.plot(possible_mu, p_mu)
        plt.legend(['Binomial p of mistake', 'p of mu'])
        plt.tight_layout()
    return np.sum(bin_over_half * p_mu)


def p_of_mistake(n_volunteers, n_mu):
    
    mu = possible_mu(n_mu)
    p_mu = p_mu_uniform(n_mu)
    p_mistake_at_mu = np.ones_like(mu)

    for i in range(n_mu):
        if mu[i] < 0.5:
            p_mistake_at_mu[i] = p_mu[i] * p_bin_over_half(n_volunteers, mu[i])
        else:
            p_mistake_at_mu[i] = p_mu[i] * p_bin_under_half(n_volunteers, mu[i])

    return p_mistake_at_mu


if __name__ == '__main__':

    n_mu = 1001
    even_n_vols = range(4, 40)[::8]
    odd_n_vols = range(5, 40)[::8]

    fig, axes = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,sharey=True)

    for n_vols in even_n_vols:
        axes[0].plot(possible_mu(n_mu), p_of_mistake(n_vols, n_mu))
    axes[0].legend(even_n_vols)
    axes[0].set_xlabel('True vote fraction')
    axes[0].set_ylabel('Prob. density of wrong binary label')
    axes[0].set_title('Even n of volunteers')

    for n_vols in odd_n_vols:
        plt.plot(possible_mu(n_mu), p_of_mistake(n_vols, n_mu))
    axes[1].legend(odd_n_vols)
    axes[1].set_xlabel('True vote fraction')
    axes[1].set_title('Odd n of volunteers')

    fig.tight_layout()
    fig.savefig('figures/n_volunteers_pdf.png')

    plt.clf()
    n_vols = [n for n in range(4, 41)]
    p_mistake = list(map(lambda x: p_of_mistake(x, 101).sum(), n_vols))
    plt.scatter(n_vols, p_mistake, marker='+')
    plt.axhline(p_mistake[-1], c='r', linestyle='--')
    plt.xlabel('Volunteers')
    plt.ylabel('P of mistaken label')
    plt.tight_layout()
    plt.savefig('figures/p_of_mistake.png')
