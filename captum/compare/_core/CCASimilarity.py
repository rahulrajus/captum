from typing import Any, Callable, List, Tuple, Union

from captum.compare.fb._core.ModuleSimilarity import ModuleSimilarity
from captum.compare.fb._utils.cca import cca
from captum.compare.fb._utils.similarity import compare_layers
from torch import Tensor, nn


class CCASimilarity(ModuleSimilarity):
    r"""
    CCA computes similarity based on dimensionality reduction similar to Principle Component Analysis (PCA). In contrary to PCA which tries to reduce the dimensionality for one vector, CCA reduces the dimensionality for a pair of vectors; instead of explaining the overall variance it explains overall correlation between those two vectors. The goal of CCA is to summarize the covariance matrix between those two vectors with p numbers where p is the canonical correlations.
    """

    def __init__(
        self,
        model1: nn.Module = None,
        model2: nn.Module = None,
        model1_id: str = "model1",
        model2_id: str = "model2",
    ):
        r"""
        Args:
            model1 (nn.Module): First model that will be used to run a comparison
            model2 (nn.Module): Second model that will be used to run a comparison
            model1_id (str): ID of the first model that will be used for loading/saving activations
            model2_id (str): ID of the second model that will be used for loading/saving activations
        """
        super().__init__(model1, model2)
        self.model1_id = model1_id
        self.model2_id = model2_id

    def compare(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]] = None,
        transform: Union[
            Callable, List[Union[Tuple[Callable, Callable], Callable]]
        ] = None,
        additional_forward_args: Any = None,
        batch_size: int = 1000,
        show_progress: bool = True,
        save_dir: str = ".",
        load_from_disk: bool = True,
        input_identifier: str = None,
        epsilon: float = 1e-10,
        threshold: float = 0.98,
        compute_directions: bool = False,
    ):
        r"""
        Function for running CCA algorithm on models. Provides utilities for layer activation transforms, algorithm parameters, and methods for comparing one layer pair or multiple layer pairs. Function can also work on standalone layers in the case where models are not defined.

        Args:
            inputs (Union[Tensor, List[Tensor, ...]]): Input that will be passed into
                both models. If either model is undefined, we expect all layers to be standalone. As a result, inputs should be the length of layer_pairs.
            layer_pairs (): List of pairs of nn modules to be compared. To compare just
                2 layers, a user can also pass in just one tuple
            transform: Structure for applying transformations to all layers, layer
                pairs, or individual layers. The interface offers several options:
                    * Callable: custom function applies to all high dimensional layers
                    * str: use an existing transform implementation (avg_pool or flatten)
                    * [(Callable | str)]: list of either custom function or existing transform. Each transform will be applied to both layers in the pair
                    * [(Callable|str, Callable|str)]: List of tuples for different transform functions for each layer in a pair. Each index in list corresponds to a layer pair.
            additional_forward_args (Any): Any additional arguments to pass into forward pass
            batch_size (int): Batch size to use with the inputs. Lower batch size
                will use less memory but requires for forward passes. A higher batch size is the fastest if you have sufficient GPU memory
            save_dir (str): Directory to save layer activations
            load_from_disk (bool): Flag to decide whether function will load existing
                activations from disk or generate again from scratch
            input_identifier (str): Unique string representing the input. This is
                important if running comparisons with different datasets. If an identifier is not set, running this function on different inputs may overwrite previously saved activations
            epsilon (float): small float to help stabilize computations
            threshold (float): float between 0, 1 used to get rid of trailing zeros in
                the cca correlation coefficients to output more accurate
                summary statistics of correlations.
            compute_directions (bool): boolean value determining whether actual cca
                directions are computed. (For very large neurons and
                datasets, may be better to compute these on the fly
                instead of store in memory). Note that this must be set to true if this similarity function will be used with the CCA UI Explorer.

        Returns:
            out (Union[Tensor, Dict[Tensor]]): Output of comparison function.
        """
        return compare_layers(
            self.model1,
            self.model2,
            self.model1_id,
            self.model2_id,
            inputs,
            layer_pairs,
            transform,
            additional_forward_args,
            batch_size=batch_size,
            show_progress=show_progress,
            save_dir=save_dir,
            load_from_disk=load_from_disk,
            input_identifier=input_identifier,
            similarity_method=cca,
            epsilon=epsilon,
            threshold=threshold,
            compute_directions=compute_directions,
        )
