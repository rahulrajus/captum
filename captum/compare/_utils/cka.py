from typing import Callable, Union

import torch
from torch import Tensor


r"""
CKA Implementation as described in Similarity of Neural Representations Revisited. Kornblith et al. (2019), https://arxiv.org/pdf/1905.00414.pdf

Code based on a Google-Research project available under Apache License 2.0
https://github.com/google-research/google-research/blob/master/LICENSE
"""

def dot(x: Tensor, y: Tensor = None):
    r"""
    Generic function for taking a dot product. If y is None, then function performs a self product with x instead.
    """
    return x.mm(x.T) if (y is None) else x.mm(y)

def linear_kernel(x: Tensor):
    r"""
    Gram matrix with a linear kernel.
    Args:
        act (Tensor): Input activation
    Returns:
        out (Tensor): Gram matrix
    """
    return dot(x)

def median_heuristic(distances: Tensor):
    return torch.median(distances)

def rbf_kernel(x: Tensor, bandwidth: float=1.0, normalization_fn: Callable=median_heuristic):
    r"""
    Gram Matrix with RBF kernel
    Args:
        x (Tensor): Input activation
        bandwidth (float): Float representing bandwidth value in RBF
        normalization_fn (Callable): Function should take in distances and output a float to scale the bandwidth. The default is set to the median of the distances as described in Kornblith et al. (2019)
    Returns:
        rbf (Tensor): Gram matrix representing the normalized rbf kernel of the input tensor
    """
    linear_mat = linear_kernel(x)
    squared_values = torch.einsum('ii->i', linear_mat)
    k1 = squared_values[:, None]
    k2 = squared_values[None, :]
    distances = k1 + k2 - 2 * linear_mat  # ||x-y||**2 =  xTx + yTy - 2xTy
    norm_factor = normalization_fn(distances)
    rbf = torch.exp(-distances / (2 * (bandwidth ** 2) * norm_factor))
    return rbf

def center(x: Tensor):
    r"""
    Given a tensor, function will output a new tensor that has been centered. The formula for generating the centering matrix is: In - (1/n)*1n
    """
    n = x.shape[0]
    centering_matrix = torch.eye(n,n) - 1.0 / n * torch.ones(n,n)
    return torch.mm(x, centering_matrix.to(x.device))

def vanilla_hsic(k: Tensor, l: Tensor):
    r"""
    Empirical HSIC equation implementation from equation 3, Kornblith et al. (2019)
    https://arxiv.org/pdf/1905.00414.pdf
    """
    k_center = center(k.float())
    l_center = center(l.float())
    khlh = torch.mm(k_center, l_center)
    hsic = (1.0/((k.shape[0]-1)**2))*torch.trace(khlh)
    return hsic

def debiased_center(x: Tensor):
    r"""
    Debiased implementation for centering activation tensors. Based on U-statistic formulation in Szekely, G. J., & Rizzo, M. L. (2014).
    """
    n = x.shape[0]
    x = x.fill_diagonal_(0)
    means = torch.sum(x, 0, dtype=torch.float64) / (n - 2)
    means -= torch.sum(means) / (2 * (n - 1))
    x -= means[:, None]
    x -= means[None, :]
    x = x.fill_diagonal_(0)
    return x

def debiased_hsic(k: Tensor, l: Tensor):
    r"""
    Debiased HSIC calculation.
    """
    k = debiased_center(k)
    l = debiased_center(l)
    return k.ravel().dot(l.ravel())

def debiased_linear_hsic(
    dot, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
    r"""Computing debiased dot product similarity (i.e. linear HSIC) based on estimator from Song et al. (2007)."""
    return (
      dot - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  r"""Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. This estimator may be negative.
  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - torch.mean(features_x, 0, keepdims=True)
  features_y = features_y - torch.mean(features_y, 0, keepdims=True)

  squared_norm_xy = torch.linalg.norm(dot(features_x.T, features_y)) ** 2
  norm_xx = torch.linalg.norm(dot(features_x.T))
  norm_yy = torch.linalg.norm(dot(features_y.T))

  if debiased:
    n = features_x.shape[0]
    sum_squared_rows_x = torch.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = torch.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = torch.sum(sum_squared_rows_x)
    squared_norm_y = torch.sum(sum_squared_rows_y)

    squared_norm_xy = debiased_linear_hsic(
        squared_norm_xy, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    norm_xx = torch.sqrt(debiased_linear_hsic(
        norm_xx ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    norm_yy = torch.sqrt(debiased_linear_hsic(
        norm_yy ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return squared_norm_xy / (norm_xx * norm_yy)

def cka(
    act1: Tensor,
    act2: Tensor,
    kernel: Union[str, Callable] = "linear",
    bandwidth: float = 1.0,
    normalization_fn: Callable = median_heuristic,
    debiased: bool = False,
    use_features: bool = False
):
    r"""
    Modification of CKA Authors' implementation of kernel based CKA. Includes utility for user defined kernel function and predefined kernels: linear & rbf.
    Args:
        act1 (Tensor): first activation tensor (must be 2D)
        act2 (Tensor): second activation tensor (must be 2D)
        kernel (Union[str, Callable]): kernel to apply to activations. User can pass in a custom kernel function that returns a gram matrix or use an existing implementation by passing in the strings 'linear' or 'rbf'.
        bandwidth (float): Float representing bandwidth value in RBF
        normalization_fn (Callable): Function should take in distances and output a float to scale the bandwidth. The default is set to the median of the distances as described in Kornblith et al. (2019). Note that is only used in the RBF kernel.
        debiased (bool): Flag to determine whether to use a debiased calculation or not
        use_features (bool): Flag to determine whether to use feature-space or example-space in CKA calculation. As described in Kornblith et al. (2019), feature space calculations use a linear kernel and are typically faster when there are more examples than features.
    Returns:
        cka_score (Tensor): tensor with CKA score
        stats (Dict[str, Tensor]): dictionary with relevant stats. Currently just includes cka_score
        epsilon (float): small float to help with stabilizing computations
    """
    assert act1.dim() == 2, "First activation should be 2D"
    assert act2.dim() == 2, "Second activaton should be 2D"

    if use_features:
        cka_val = feature_space_linear_cka(act1, act2, debiased=debiased)
    else:
        if isinstance(kernel, Callable):
            try:
                act1 = kernel(act1)
                act2 = kernel(act2)
            except Exception:
                raise RuntimeError("There was a problem executing your function")
            assert act1.shape[0] == act1.shape[1], "kernel must return square gram matrix"
            assert act2.shape[0] == act2.shape[1], "kernel must return square gram matrix"

        elif kernel == "linear":
            act1 = linear_kernel(act1)
            act2 = linear_kernel(act2)
        elif kernel == "rbf":
            act1 = rbf_kernel(act1, bandwidth=bandwidth, normalization_fn=normalization_fn)
            act2 = rbf_kernel(act2, bandwidth=bandwidth, normalization_fn=normalization_fn)
        if(debiased):
            hsic_kl = debiased_hsic(act1, act2)
            hsic_k = debiased_hsic(act1, act1)
            hsic_l = debiased_hsic(act2, act2)
        else:
            hsic_kl = vanilla_hsic(act1, act2)
            hsic_k = vanilla_hsic(act1, act1)
            hsic_l = vanilla_hsic(act2, act2)

        cka_val = hsic_kl/torch.sqrt(hsic_k*hsic_l)

    stats = {"similarity": cka_val}
    return cka_val, stats
