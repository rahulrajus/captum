from typing import Any, List, Tuple, Union

import torch
from torch import Tensor, nn

from captum._utils.fb.av import AV


def layer_attributions(
    model: nn.Module,
    layers: List[Union[nn.Module, str]],
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    additional_forward_args: Any = None,
    save_dir: str = ".",
    model_id: str = "net",
    identifier: str = None,
    load_from_disk: bool = True,
    batch_size: int = 1024,
):
    r"""
    Function for getting activations from model layers. User can either supply a single layer or a list of layers. The input is passed into both model1 and model2 and a forward pass is made to retrive activations. Function returns the layer activations for each layer that is passed in.

    Args:
        model (nn.Module): Model where layers are stored. Function will use this object to get string representations of all the layers passed in & pass in the inputs to retrieve layer activations.
        layers(List[nn.Module]): List of layers to retrieve activations from
        inputs (Union[Tensor, Tuple[Tensor, ...]]): Input to be passed into the model
        additional_forward_args (Any): Any additional arguments required for the forward pass.
        save_dir (str): Location to store activations for loading later
        model_id (str): Identifier for model being passed in
        identifier (str): An optional identifier for the layer activations. Typically used for storing multiple batches of the same layer activation
        load_from_disk (bool): Forces function to regenerate activations if False
    Returns:
        attributions (Dict[str, Tensor]): Dictionary of layer activations where the key is the string representation of the layer & the value is the activation tensor
    """

    unsaved_layers = []

    if load_from_disk:
        for _, layer in enumerate(layers):
            if not AV.exists(save_dir, model_id, layer, identifier):
                unsaved_layers.append(layer)
    else:
        unsaved_layers = layers

    # if user decides to explicitly pass in None for batch_size, set batch_size to length of the input. Otherwise function will use default value of 1024 (as defined in the batch_size argument)
    if not batch_size:
        batch_size = inputs.shape[0]

    inputs = torch.split(inputs, batch_size)
    if len(unsaved_layers) > 0:
        for i, inpt_batch in enumerate(inputs):
            if identifier:
                identifier_batch = identifier + "-" + str(i * batch_size)
            else:
                identifier_batch = str(i * batch_size)
            AV.generate_activation(
                save_dir,
                model,
                model_id,
                unsaved_layers,
                inpt_batch,
                identifier_batch,
                additional_forward_args,
            )
    model_attributions = {}
    for layer in set(layers):
        if not AV.exists(save_dir, model_id, layer, identifier):
            raise RuntimeError(f"Layer {layer} was not found in manifold")
        else:
            layer_attr = []
            for attr in AV.load(save_dir, model_id, layer, identifier):
                layer_attr.append(attr.squeeze(0))
            layer_attr = torch.cat(layer_attr)
            model_attributions[layer] = layer_attr

    return model_attributions
