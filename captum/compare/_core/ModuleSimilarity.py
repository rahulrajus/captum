from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Union

from torch import Tensor, nn


class ModuleSimilarity(ABC):
    r"""
    An abstract class to define module similarity skeleton.
    """

    def __init__(
        self, model1: nn.Module = None, model2: nn.Module = None, **kwargs: Any
    ):
        self.model1 = model1
        self.model2 = model2

    @abstractmethod
    def compare(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]] = None,
        transform: Union[
            Callable, List[Union[Tuple[Callable, Callable], Callable]]
        ] = None,
        additional_forward_args: Any = None,
        batch_size: int = 1024,
        save_dir: str = ".",
        load_from_disk: bool = True,
        **kwargs: Any
    ) -> Union[Tensor, Dict]:
        r"""
        Abstract method for representation similarity algorithm. Goal of method is to run any model similarity algorithm across user specified layer pairs and return relevant metrics and statistics.
        """
        pass
