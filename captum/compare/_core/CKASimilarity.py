from typing import Any, Callable, List, Tuple, Union

from torch import Tensor, nn

from captum.compare.fb._core.ModuleSimilarity import ModuleSimilarity
from captum.compare.fb._utils.cka import cka, median_heuristic
from captum.compare.fb._utils.similarity import compare_layers


class CKASimilarity(ModuleSimilarity):
    r"""
    CKA similarity computes similarity metric using Hilbert-Schmidt independence criterion using linear or rbf kernels. Includes several customization utilities for more granular access to how the comparison is done across different layer types.
    """

    def __init__(
        self,
        model1: nn.Module = None,
        model2: nn.Module = None,
        # TODO: Move to abstract class?
        model1_id: str = "model1",
        model2_id: str = "model2",
    ):
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
        batch_size: int = 1024,
        show_progress: bool = True,
        save_dir: str = ".",
        load_from_disk: bool = True,
        input_identifier: str = None,
        kernel: Union[str, Callable] = "linear",
        bandwidth: float = 1.0,
        normalization_fn: Callable = median_heuristic,
        debiased: bool = False,
        use_features: bool = False
    ):
        r"""
        Function for running CKA algorithm on models. Provides utilities for layer activation transforms, algorithm parameters, and methods for comparing one layer pair or multiple layer pairs. Function can also work on standalone layers in the case where models are not defined.

        Args:
            model1 (nn.Module): First model
            model2 (nn.Module): Second model
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
            batch_size (int): Batch size to use for splitting the input. Inputs will be batched before passing into model1 and model2. After all forward passes are complete, resulting activations will be merged.
            show_progress (bool): Flag to determine whether to display a progress bar for cka calculations. If set to true, a progress bar will display how many layer comparisons have been completed.
            save_dir (str): Directory to save layer activations
            load_from_disk (bool): Flag to decide whether function will load existing
                activations from disk or generate again from scratch
            input_identifier (str): Unique string representing the input. This is
                important if running comparisons with different datasets. If an identifier is not set, running this function on different inputs may overwrite previously saved activations
            kernel (Callable | str): kernel to use to compute HSIC. Takes in default
                implementations of 'linear' and 'rbf' as strings. Can also take in a custom kernel function
            bandwidth (float): Float representing bandwidth value in RBF
            normalization_fn (Callable): Function should take in distances and output a float to scale the bandwidth. The default is set to the median of the distances as described in Kornblith et al. (2019). Note that is only used in the RBF kernel.
            debiased (bool): Flag to determine whether to use a debiased calculation
            use_features (bool): Flag to determine whether to use feature-space or example-space in CKA calculation. As described in Kornblith et al. (2019), feature space calculations use a linear kernel and are typically faster when there are more examples than features.

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
            similarity_method=cka,
            kernel=kernel,
            bandwidth=bandwidth,
            normalization_fn=normalization_fn,
            debiased=debiased,
            use_features=use_features
        )
