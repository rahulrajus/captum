from collections.abc import Callable
from types import FunctionType

import torch
from torch import Tensor


def avg_pool_4d(act: Tensor) -> Tensor:
    r"""
    Utility for average pooling on 4D activation vectors. Since users may use this as a default transform function, function will return original tensor unchanged if it is already 2D

    Args:
        act (Tensor): Activation vector to be transformed

    Returns:
        transformed_act (Tensor): Average pooled tensor
    """
    if act.dim() == 4:
        return torch.mean(act, axis=(2, 3))
    elif act.dim() == 2:
        print(
            "Input activation is not 4D. Since activation is already 2D, vector will be returned unchanged"
        )
        return act
    raise RuntimeError(
        "Input dimensions not valid. This function only works on 4D Tensors"
    )


def flatten(act: Tensor) -> Tensor:
    r"""
    Utility for flattening activation vector.

    Args:
        act (Tensor): Activation vector to be transformed

    Returns:
        transformed_act (Tensor): Flattened tensor
    """

    return act.reshape((len(act), -1))


def transform_activation(act: Tensor, transform: Callable):
    r"""
    Generic utility function for pooling/downsampling an activation tensor. Function includes implementations for average pooling and flattening to reduce 4D tensors to 2D tensors that can be supplied to a comparison algorithm ((SV)CCA, PWCCA, CKA, etc.). Function also provides ability for user to pass in a custom function.

    Args:
        act (Tensor): Activation vector that will be transformed
        transform (transform: Callable): Method for transforming
                    activation. User is expected to pass in a function that will be applied to the activation tensor.
    Returns:
        transformed_act (Tensor): A transformed version of the passed in activation
                        vector. If a custom transform was passed in, it must return a 2D tensor otherwise the function will raise a runtime error.
    """
    if not transform:
        if act.dim() == 2:
            return act
        else:
            return flatten(act)  # provide flatten as default transform
    if isinstance(transform, FunctionType):
        try:
            out = transform(act)
            if out.dim() != 2:
                raise RuntimeError(
                    "Activation transform must return a 2 dimensional tensor"
                )
            return out
        except Exception:
            raise RuntimeError("Transform function not valid!")
    else:
        raise RuntimeError("Transform argument must be a function")
