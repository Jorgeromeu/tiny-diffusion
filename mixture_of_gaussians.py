import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal


def random_isotropic_mog(
    num_modes,
    dim,
    mean_lo=-1.5,
    mean_hi=1.5,
    scale_lo=0.05,
    scale_hi=0.1,
    generator=None,
):
    means = torch.empty(num_modes, dim).uniform_(mean_lo, mean_hi, generator=generator)
    scales = torch.empty(num_modes).uniform_(scale_lo, scale_hi, generator=generator)

    covariances = torch.stack([torch.eye(dim) * scale for scale in scales])

    # uniform weights
    weights = torch.ones(num_modes)
    weights /= weights.sum()

    return construct_mog(weights, means, covariances)


def construct_mog(weights, means, covariances):
    categorical = Categorical(weights)
    gaussians = MultivariateNormal(loc=means, covariance_matrix=covariances)
    mog = MixtureSameFamily(
        mixture_distribution=categorical, component_distribution=gaussians
    )
    return mog


def construct_isotropic_mog(weights, means, scales):
    covariances = torch.stack([torch.eye(means.size(-1)) * scale for scale in scales])
    return construct_mog(weights, means, covariances)


def diffuse_mog(mog, alphas_cumprod, t):
    mixture_dist = mog.mixture_distribution
    component_dist = mog.component_distribution
    means = component_dist.loc
    covariances = component_dist.covariance_matrix

    alpha_t = alphas_cumprod[t]

    diffused_means = torch.sqrt(alpha_t) * means

    dim = means.size(-1)

    identities = torch.eye(dim).expand(means.size(0), dim, dim)
    diffused_covs = alpha_t * covariances + (1 - alpha_t) * identities

    diffused_mog = torch.distributions.MultivariateNormal(diffused_means, diffused_covs)
    diffused_mog = torch.distributions.MixtureSameFamily(mixture_dist, diffused_mog)
    return diffused_mog
