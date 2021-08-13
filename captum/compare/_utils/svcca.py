from captum.compare.fb._utils.cca import subarray_by_sum, get_cca_similarity
import torch


def svcca(acts1, acts2, threshold=0.99, epsilon=1e-10, compute_directions=False):
    r"""
    Modification of SVCCA Authors' implementation.
    Args:
        act1 (Tensor): first activation tensor (num_examples, num_features)
        act2 (Tensor): second activation tensor (num_examples, num_features)
        threshold (float): Used to determine how many singular directions should be used in the cca calculation
        epsilon (float): small float to help with stabilizing computations
    Returns:
        svcca_score (Tensor): tensor with SVCCA score
        stats (Dict[str, Tensor]): dictionary with relevant stats. Includes directions, variates, and mean statistics. More specifically, dictionary includes neuron
        coefficients (including full coefficients), inverse square roots of sigma xx and yy, CCA coefficients, CCA directions for both activations, indices from x and y that were removed, and the CCA similarity score

    """
    # CCA expects (feature_dim, examples)
    acts1 = acts1.T
    acts2 = acts2.T

    cacts1 = acts1 - torch.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - torch.mean(acts2, axis=1, keepdims=True)
    # Perform SVD
    U1, s1, V1 = torch.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = torch.linalg.svd(cacts2, full_matrices=False)

    s1_dim = subarray_by_sum(s1, threshold)

    s2_dim = subarray_by_sum(s2, threshold)

    svacts1 = torch.matmul(
        s1[: s1_dim ] * torch.eye(s1_dim ).to(acts1.device), V1[: s1_dim ]
    )
    svacts2 = torch.matmul(
        s2[: s2_dim ] * torch.eye(s2_dim ).to(acts1.device), V2[: s2_dim ]
    )
    svcca_results = get_cca_similarity(svacts1, svacts2, epsilon=epsilon, compute_directions=compute_directions)

    return svcca_results["similarity"], svcca_results
