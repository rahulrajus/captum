import itertools
from types import FunctionType
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor, nn

from captum._utils.progress import progress
from captum.compare.fb._utils.activations import layer_attributions
from captum.compare.fb._utils.transform import transform_activation


# TODO: Move to common location?
def _modules_to_string(model: nn.Module, layers: List[nn.Module]):
    # Converts a list of nn.Module layers to their string representation
    model_lookup = {v: k for k, v in model.named_modules()}
    return [model_lookup[layer] for layer in layers]


# TODO: Move to common location?
def _model_children(model: nn.Module):
    # Get all children from model. Based on https://stackoverflow.com/a/65112132
    children = []
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            children.append(name)
    return children


# TODO: Move to common location?
def _permutate_lists(vec1: List, vec2: List):
    return list(itertools.product(vec1, vec2))


def _get_layer_permutations(
    model1: nn.Module, model2: nn.Module
) -> List[Tuple[nn.Module, nn.Module]]:
    if not model1 or not model2:
        raise RuntimeError("Both models must be provided if layer pairs is undefined")
    model1_layers = _model_children(model1)
    model2_layers = _model_children(model2)
    layer_pairs = _permutate_lists(model1_layers, model2_layers)
    return layer_pairs


def _get_standalone_layer_activations(
    layer_pairs: List[Tuple[nn.Module, nn.Module]], inputs: List[Tensor]
) -> Tuple[List[Tensor], List[Tensor]]:
    r"""
    In the case where user passes in standalone activations (no model was passed in), this utility function is used to loop over the passed in layer_pairs and run a forward pass on each pair with the corresponding input. Keep in mind that in this case, the length of layer_pairs and inputs must be exactly the same. Function will pass in inputs[i] into layer_pairs[i].

    Args:
        layer_pairs (List[Tuple[nn.Module, nn.Module]]): List of standalone layer pairs. Each pair consists of 2 nn.Modules. Both modules in a pair will have the same input passed into them.
        inputs (List[Tensor]): List of inputs to be passed into each layer pair. Each index represents 1 input that will be passed into the corresponding layer pair.

    Returns:
        Tuple[List[Tensor], List[Tensor]]: [description]
    """
    if len(inputs) != len(layer_pairs):
        raise RuntimeError(
            "If models are not provided, length of inputs must match the number of pairs"
        )
    pair1_attributions = []
    pair2_attributions = []
    for i in range(len(layer_pairs)):
        layer1, layer2 = layer_pairs[i]
        act1 = layer1(inputs[i])
        act2 = layer2(inputs[i])
        pair1_attributions.append(act1)
        pair2_attributions.append(act2)
    return pair1_attributions, pair2_attributions


def compare_layer(
    act1: Tensor,
    act2: Tensor,
    transform1: Union[Callable] = None,
    transform2: Union[Callable] = None,
    similarity_method: Callable = None,
    **kwargs
):

    r"""
    Generic utility function for comparing 2 activation vectors using a user-passed in comparison function. User is responsible for supplying a comparison function to use for comparing activations. There are also optional transform functions that can be used for pooling/downsampling activations (typically used for vectors with high dimensionality, such as convolutional layers)

    Args:
        act1 (Tensor): First activation tensor
        act2 (Tensor): Second activation tensor.
        transform1 (Union[str, Callable]): Transform function for first activation
                    vector. User can also pass in 'avg_pool' or 'flatten' to use our implementations.
        transform2 (Union[str, Callable]): Transform function for second activation.
                    Follows same rules as transform1.
        similarity_method (Callable): User must pass in the desired function to use for
                    comparing transformed activation vectors. Additional args that are needed by the function must be passed in through keyword arguments (**kwargs)
    Returns:
        out (Union[Tensor, Dict[Tensor]]): Output of comparison function.

    """
    # apply user's transformations
    act1 = transform_activation(act1, transform1)
    act2 = transform_activation(act2, transform2)
    if act1.dim() != 2:
        raise RuntimeError(
            "First activation must be transformed to a 2D tensor of shape (num_samples, feature dimensions"
        )
    if act2.dim() != 2:
        raise RuntimeError(
            "Second activation must be transformed to a 2D tensor of shape (num_samples, feature dimensions"
        )
    try:
        score, stats = similarity_method(act1, act2, **kwargs)
    except Exception:
        raise RuntimeError(
            "There was an issue with your comparison function. Please make sure your function returns both a score and a dictionary with stats (or None if N/A)"
        )
    return score, stats


def compare_layers(
    model1: nn.Module,
    model2: nn.Module,
    model1_id: str,
    model2_id: str,
    inputs: Union[Tensor, List[Tensor]],
    layer_pairs: Union[
        Tuple[str, str],
        Tuple[Callable, Callable],
        List[Tuple[str, str]],
        List[Tuple[Callable, Callable]],
    ],
    transform: Union[Callable, List[Union[Tuple[Callable, Callable], Callable]]],
    additional_forward_args: Any,
    batch_size: int,
    save_dir: str,
    load_from_disk: bool,
    input_identifier: str,
    show_progress: bool,
    similarity_method: Callable,
    **kwargs
) -> Tuple[Tensor, Dict]:
    r"""
    Generic function for running a similarity algorithm on models. User can pass in their own similarity function and can pass in any required parameters via keyword arguments.

    Args:
        model1 (nn.Module): First model
        model2 (nn.Module): Second model
        inputs (Union[Tensor, List[Tensor]]): Input that will be passed into both models. If either model is undefined, we expect all layers to be standalone. As a result, inputs should be the length of layer_pairs.
        layer_pairs (): List of pairs of nn modules to be compared. To compare just 2 layers, a user can also pass in just one tuple
        transform: Structure for applying transformations to all layers, layer pairs,
        or individual layers. The interface offers several options:
            * Callable: Transform function will be applied to all layers if possible
            * [Callable]: List of tuples
            * str: use an existing transform implementation (avg_pool or flatten)
            * [(Callable)]: list of either custom function transforms. Each transform will be applied to both layers in the pair
            * [(Callable, Callable)]: List of tuples for different transform functions for each layer in a pair. Each index in list corresponds to a layer pair.
        additisonal_forward_args (Any): Any additional arguments to pass into forward pass for either model
        batch_size (int): Applies batching to inputs. Keep in mind that batching can reduce speed since it required multiple forward passes on both models!
        save_dir (str): Directory to save activations
        load_from_disk (bool): Flag to decide whether function will load existing activations from disk or generate again from scratch
        input_identifier (str): Unique string representing the input. This is important if running comparisons with different datasets. If an identifier is not set, running this function on different inputs may overwrite previously saved activations.
        show_progress (bool): Flag to determine whether to display a progress bar for cka calculations. If set to true, a progress bar will display how many layer comparisons have been completed
        similarity_method (Callable): Similarity function that will be used for making layer comparisons
    Returns:
        out (Union[Tensor, Dict[Tensor]]): Output of comparison function.
    """
    # if layer pairs is None, we should get permutations of all layers!
    if not layer_pairs:
        layer_pairs = _get_layer_permutations(model1, model2)
    if not isinstance(layer_pairs, list):
        layer_pairs = [layer_pairs]
    if isinstance(transform, list) and len(transform) != len(layer_pairs):
        raise RuntimeError("Transform list must match length of layer_pairs")

    model1_layers, model2_layers = zip(*layer_pairs)
    model1_attributions = []
    model2_attributions = []
    use_standalone = False
    if model1 and model2:
        model1_attributions = layer_attributions(
            model1,
            model1_layers,
            inputs,
            additional_forward_args,
            model_id=model1_id,
            save_dir=save_dir,
            load_from_disk=load_from_disk,
            identifier=input_identifier,
            batch_size=batch_size,
        )

        model2_attributions = layer_attributions(
            model2,
            model2_layers,
            inputs,
            additional_forward_args,
            model_id=model2_id,
            save_dir=save_dir,
            load_from_disk=load_from_disk,
            identifier=input_identifier,
            batch_size=batch_size,
        )

    else:  # if a model is not defined, then we are expecting standalone layers
        model1_attributions, model2_attributions = _get_standalone_layer_activations(
            layer_pairs, inputs
        )
        use_standalone = True

    similarity_list = []
    stats: Dict[nn.Module, Dict[nn.Module, Dict[str, Tensor]]] = {}
    len_pairs = len(model1_layers)
    range_pairs = range(len_pairs)
    if show_progress:
        range_pairs = progress(range_pairs)
    for i in range_pairs:
        if use_standalone:
            act1 = model1_attributions[i]
            act2 = model2_attributions[i]
        else:
            act1 = model1_attributions[model1_layers[i]]
            act2 = model2_attributions[model2_layers[i]]

        # Code for processing transform types
        transform1 = None
        transform2 = None

        if isinstance(transform, list):
            transform_func = transform[i]
            if isinstance(transform_func, tuple):
                transform1 = transform_func[0]
                transform2 = transform_func[1]
            elif isinstance(transform_func, FunctionType):
                transform1 = transform_func
                transform2 = transform_func
            else:
                raise ValueError(
                    "Transform argument invalid. Expecting tuple or Callable. Refer to docstring for more information!"
                )
        elif not transform or isinstance(transform, FunctionType):
            transform1 = transform
            transform2 = transform
        else:
            raise ValueError(
                "Transform argument invalid. Must be of type List or Callable"
            )
        out, pair_stats = compare_layer(
            act1, act2, transform1, transform2, similarity_method, **kwargs
        )

        # Generate similarity and stats data
        similarity_list.append(out)
        layer1 = model1_layers[i]
        layer2 = model2_layers[i]
        if layer1 not in stats:
            stats[layer1] = {}
        stats[layer1][layer2] = pair_stats

    similarity_list = torch.tensor(similarity_list)
    return similarity_list, stats
