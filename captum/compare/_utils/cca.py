import torch
from torch import Tensor


def cov(tensor, rowvar=True, bias=False):
    r"""
    Estimate covariance matrix. Based on implementation in
    https://github.com/pytorch/pytorch/issues/19037
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def subarray_by_sum(array, threshold):
    r"""
    Helper function that computes threshold index of a nonnegative array by summing.
    This function takes in an array of nonnegative floats, and a
    threshold between 0 and 1. It returns the index i at which the sum of the
    array up to i is greater than or equal to our threshold
    Args:
              array: a 1d torch tensor of nonnegative floats
              threshold: a number between 0 and 1
    Returns:
              i: index at which cumulative sum >= threshold
    """
    assert (threshold >= 0) and (threshold <= 1), "incorrect threshold"

    arr_sum = torch.sum(array)
    curr_sum = 0.0
    for i, num in enumerate(array):
        curr_sum += num
        if curr_sum / arr_sum >= threshold:
            return i


"""
The following source code is adapted from:
https://github.com/google/svcca
Code is modified for PyTorch support
Licensed under the Apache License, Version 2.0
"""


def positivedef_matrix_sqrt(array: Tensor):
    r"""
    Stable method for computing matrix square roots, supports complex matrices.
    Args:
              array: torch 2d tensor, can be complex valued that is a positive
                     definite symmetric (or hermitian) matrix
    Returns:
              sqrtarray: The matrix square root of array
    """
    w, v = torch.linalg.eigh(array)
    wsqrt = torch.sqrt(torch.abs(w))
    sqrtarray = torch.matmul(v, torch.matmul(torch.diag(wsqrt), torch.conj(v).T))
    return sqrtarray


def compute_ccas(sigma_xx: Tensor, sigma_xy: Tensor, sigma_yy: Tensor, epsilon: float):
    r"""
    Main cca computation function, takes in variances and crossvariances.
    This function takes in the covariances and cross covariances of the inputs being compared, preprocesses them (removing small magnitudes) and outputs the raw results of the cca computation, including cca directions in a rotated space, and the
    cca correlation coefficient values.
    Args:
              sigma_xx: 2d torch tensor, (num_neurons_x, num_neurons_x)
                        variance matrix for x
              sigma_xy: 2d torch tensor, (num_neurons_x, num_neurons_y)
                        crossvariance matrix for x,y
              sigma_yy: 2d torch tensor, (num_neurons_y, num_neurons_y)
                        variance matrix for y
              epsilon:  small float to help with stabilizing computations by adding noise along the diagonal and getting rid of zeros.
    Returns:
              [ux, sx, vx]: [torch 2d tensor, torch 1d tensor, torch 2d tensor]
                            ux and vx are (conj) transposes of each other, being
                            the canonical directions in the X subspace.
                            sx is the set of canonical correlation coefficients-
                            how well corresponding directions in vx, Vy correlate
                            with each other.
              [uy, sy, vy]: Same as above, but for Y space
              invsqrt_xx:   Inverse square root of sigma_xx to transform canonical
                            directions back to original space
              invsqrt_yy:   Same as above but for sigma_yy
              x_idxs:       The indexes of the input sigma_xx that were pruned
                            by remove_small
              y_idxs:       Same as above but for sigma_yy
    """

    # remove small values from covariances
    x_diag = torch.abs(torch.diagonal(sigma_xx))
    y_diag = torch.abs(torch.diagonal(sigma_yy))
    x_idxs = x_diag >= epsilon
    y_idxs = y_diag >= epsilon

    sigma_xx = sigma_xx[x_idxs][:, x_idxs]
    sigma_xy = sigma_xy[x_idxs][:, y_idxs]
    sigma_yy = sigma_yy[y_idxs][:, y_idxs]

    numx = sigma_xx.shape[0]
    numy = sigma_yy.shape[0]

    # case where all values are less than epsilon
    if numx == 0 or numy == 0:
        return (
            [0, 0, 0],
            [0, 0, 0],
            torch.zeros_like(sigma_xx),
            torch.zeros_like(sigma_yy),
            x_idxs,
            y_idxs,
        )

    numx_identity = torch.eye(numx, device=sigma_xx.device)
    numy_identity = torch.eye(numy, device=sigma_xx.device)

    sigma_xx += epsilon * numx_identity
    sigma_yy += epsilon * numy_identity

    # core cca computation (https://arxiv.org/pdf/1806.05759.pdf, page 2)

    # calculate psuedo inverse
    inv_xx = torch.linalg.pinv(sigma_xx)
    inv_yy = torch.linalg.pinv(sigma_yy)

    # calculate square root of cov xx & yy
    invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
    invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

    # multiply together
    arr = torch.matmul(invsqrt_xx, torch.matmul(sigma_xy, invsqrt_yy))

    # get cca solutions with svd (singular matrix will contain cca coefficents)
    u, s, v = torch.linalg.svd(arr)

    return [u, torch.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def get_direction(acts: Tensor, cca_coef: Tensor, invsqrt_cov: Tensor):
    r"""
    Helper function calculates a cca direction using the activation, the cca coefficient, and the inverse sqrt covariance matrix (either sigma_xx or sigma_yy).
    User is responsible for preprocessing tensors to remove noise/small values.

    Returns:
        cca_direction (Tensor): CCA direction with shape (1, acts.shape[1])
    """

    act_mean = torch.mean(acts, axis=1, keepdims=True)
    cca_direction = (
        torch.matmul(
            torch.matmul(cca_coef, invsqrt_cov),
            (acts - act_mean),
        )
        + act_mean
    )
    return cca_direction


def get_cca_similarity(
    acts1: Tensor,
    acts2: Tensor,
    epsilon: float = 0.0,
    threshold: float = 0.98,
    compute_directions: bool = True,
):
    r"""The main function for computing cca similarities.
    This function computes the cca similarity between two sets of activations,
    returning a dict with the cca coefficients, a few statistics of the cca
    coefficients, and (optionally) the actual directions.
    Args:
              acts1: (num_neurons1, data points) a 2d torch tensor of neurons by
                     datapoints where entry (i,j) is the output of neuron i on
                     datapoint j.
              acts2: (num_neurons2, data points) same as above, but (potentially)
                     for a different set of neurons. Note that acts1 and acts2
                     can have different numbers of neurons, but must agree on the
                     number of datapoints
              epsilon: small float to help stabilize computations
              threshold: float between 0, 1 used to get rid of trailing zeros in
                         the cca correlation coefficients to output more accurate
                         summary statistics of correlations.
              compute_directions: boolean value determining whether actual cca
                             directions are computed. (For very large neurons and
                             datasets, may be better to compute these on the fly
                             instead of store in memory). Note that this must be set to true if this similarity function will be used with the CCA UI Explorer.
    Returns:
              stats: A dictionary with outputs from the cca computations.
                           Contains neuron coefficients (combinations of neurons
                           that correspond to cca directions), the cca correlation
                           coefficients (how well aligned directions correlate),
                           x and y idxs (for computing cca directions on the fly
                           if compute_directions=False), and summary statistics. If
                           compute_directions=True, the cca directions are also
                           computed.
    """

    # assert dimensionality equal
    assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
    # check that acts1, acts2 are transposition
    assert acts1.shape[0] < acts1.shape[1], (
        "input must be number of neurons" "by datapoints"
    )

    stats = {}

    numx = acts1.shape[0]
    numy = acts2.shape[0]

    covariance = cov(torch.cat((acts1, acts2), axis=0))
    sigmaxx = covariance[:numx, :numx]
    sigmaxy = covariance[:numx, numx:]
    sigmayx = covariance[numx:, :numx]
    sigmayy = covariance[numx:, numx:]

    # rescale covariance to make cca computation more stable
    xmax = torch.max(torch.abs(sigmaxx))
    ymax = torch.max(torch.abs(sigmayy))
    sigmaxx /= xmax
    sigmayy /= ymax
    sigmaxy /= torch.sqrt(xmax * ymax)
    sigmayx /= torch.sqrt(xmax * ymax)

    ([u, s, v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs) = compute_ccas(
        sigmaxx, sigmaxy, sigmayy, epsilon=epsilon
    )

    # if x_idxs or y_idxs is all false, stats has zero entries
    if (not torch.any(x_idxs)) or (not torch.any(y_idxs)):
        feature_dim = acts1.shape[1]
        stats["mean"] = torch.tensor(0)
        stats["sum"] = (torch.tensor(0), torch.tensor(0))
        stats["cca_coef1"] = torch.tensor(0)
        stats["idx1"] = 0
        stats["idx2"] = 0
        stats["cca_directions1"] = torch.zeros((1, feature_dim))
        stats["cca_directions2"] = torch.zeros((1, feature_dim))
        stats["similarity"] = torch.tensor(0)
        return stats

        # also compute full coefficients over all neurons
    x_mask = torch.matmul(
        x_idxs.reshape((-1, 1)).float(), x_idxs.reshape((1, -1)).float()
    ).bool()
    y_mask = torch.matmul(
        y_idxs.reshape((-1, 1)).float(), y_idxs.reshape((1, -1)).float()
    ).bool()

    # calculate mean based on cca coefficients that explain threshold% of variance
    idx_cutoff = subarray_by_sum(s, threshold)
    adjusted_mean_similarity = torch.mean(s[:idx_cutoff])

    stats["coef_x"] = u.T.float()
    stats["invsqrt_xx"] = invsqrt_xx.float()
    # this data can be used for further calculations --> store on original device
    stats["full_coef_x"] = torch.zeros((numx, numx), device=acts1.device)
    stats["full_coef_x"].masked_scatter_(x_mask, stats["coef_x"])

    stats["full_invsqrt_xx"] = torch.zeros((numx, numx), device=acts1.device)
    stats["full_invsqrt_xx"].masked_scatter_(x_mask, stats["invsqrt_xx"])

    stats["coef_y"] = v.float()
    stats["invsqrt_yy"] = invsqrt_yy.float()

    stats["full_coef_y"] = torch.zeros((numy, numy), device=acts1.device)
    stats["full_coef_y"].masked_scatter_(y_mask, stats["coef_y"])

    stats["full_invsqrt_yy"] = torch.zeros((numy, numy), device=acts1.device)
    stats["full_invsqrt_yy"].masked_scatter_(y_mask, stats["invsqrt_yy"])

    stats["cca_coef"] = s
    stats["similarity"] = adjusted_mean_similarity
    stats["x_idxs"] = x_idxs
    stats["y_idxs"] = y_idxs
    stats["mean"] = adjusted_mean_similarity

    if compute_directions:
        stats["cca_directions1"] = get_direction(
            acts1, stats["full_coef_x"], stats["full_invsqrt_xx"]
        )
        stats["cca_directions2"] = get_direction(
            acts2, stats["full_coef_y"], stats["full_invsqrt_yy"]
        )
    return stats


def cca(
    acts1,
    acts2,
    epsilon: float = 0.0,
    threshold: float = 0.98,
    compute_directions: bool = True,
):

    stats = get_cca_similarity(acts1.T, acts2.T, epsilon, threshold, compute_directions)

    return stats["similarity"], stats
