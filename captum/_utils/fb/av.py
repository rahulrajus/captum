#!/usr/bin/env python3

import glob
import os
import re
import warnings
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

import captum._utils.common as common
from captum.attr import LayerActivation


class AV:
    r"""
    This class provides functionality to store and load activation vectors
    generated for pre-defined neural network layers.
    It also provides functionality to check if activation vectors already
    exist in the manifold and other auxiliary functions.

    This class also defines a torch `Dataset`, representing Activation Vectors,
    which enables lazy access to activation vectors and layer stored in the manifold.

    """

    r"""
        The name of the subfolder in the manifold where the activation vectors
        are stored.
    """
    AV_DIR_NAME: str = "av"

    def __init__(self) -> None:
        pass

    @staticmethod
    def _assemble_model_dir(path: str, model_id: str) -> str:
        r"""
        Returns a directory path for the given source path `path` and `model_id.`
        This path is suffixed with the '/' delimiter.
        """
        return "/".join([path, AV.AV_DIR_NAME, model_id, ""])

    @staticmethod
    def _assemble_file_path(
        source_dir: str, layer: str, identifier: Optional[str] = None
    ) -> str:
        r"""
        Returns a full filepath given a source directory, layer, and optional
        identifier. The source dir is not required to end with a "/" delimiter. The
        identifier in the output filepath is always preceded by the "-" delimiter.
        """
        if not source_dir.endswith("/"):
            source_dir += "/"
        filepath = source_dir + layer

        if identifier is not None:
            filepath = "-".join([filepath, identifier])

        return filepath

    @staticmethod
    def _construct_file_search(
        source_dir: str,
        model_id: str,
        layer: Optional[str] = None,
        identifier: Optional[str] = None,
    ) -> str:
        r"""
        Returns a search string that can be used by glob to search `source_dir/model_id`
        for the desired layer/id pair. Leaving `layer` as None will search ids over all
        layers, and leaving `identifier` as none will search layers over all ids.
        Leaving both as none will return a path to glob for every activation.

        The `identifier` argument can be an incomplete id. The returned search path will
        glob for files which include the given string in its identifier.
        """
        av_filesearch = AV._assemble_model_dir(source_dir, model_id)
        av_filesearch += layer if layer is not None else "*"
        if identifier is not None:
            av_filesearch = "-".join([av_filesearch, "*" + identifier])
        av_filesearch += "*"
        return av_filesearch

    @staticmethod
    def exists(
        path: str,
        model_id: str,
        layer: Optional[str] = None,
        identifier: Optional[str] = None,
    ) -> bool:
        r"""
        Verifies whether the model + layer activations exist
        under the manifold path.

        Args:
            path (str): The manifold path where the activation vectors
                    for the `model_id` are stored.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            layer (str or None): The layer for which the activation vectors are
                    computed.
            identifier (str or None): An optional identifier for the layer activations.
                    Can be used to distinguish between activations for different
                    training batches. For example, the id could be a suffix composed of
                    a train/test label and numerical value, such as "-train-xxxxx".
                    The numerical id is often a monotonic sequence taken from datetime.

        Returns:
            exists (bool): Indicating whether the activation vectors for the `layer`
                    and `identifier` (if provided) were stored in the manifold. If no
                    `identifier` is provided, will return `True` if any layer activation
                    exists, whether it has an identifier or not, and vice-versa.
        """
        av_dir = AV._assemble_model_dir(path, model_id)
        av_filesearch = AV._construct_file_search(path, model_id, layer, identifier)
        return os.path.exists(av_dir) and len(glob.glob(av_filesearch)) > 0

    @staticmethod
    def save(
        path: str,
        model_id: str,
        layers: Union[str, List[str]],
        act_tensors: Union[Tensor, List[Tensor]],
        identifier: Optional[str] = None,
    ) -> None:
        r"""
        Saves the activation vectors `act_tensor` for the
        `layer` under the manifold `path`.

        Args:
            path (str): The manifold path where the activation vectors
                    for the `layer` are stored.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            layers (str or List of str): The layer(s) for which the activation vectors
                    are computed.
            act_tensors (Tensor or List of Tensor): A batch of activation vectors.
                    This must match the dimension of `layers`.
            identifier (str or None): An optional identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches. For example, the id could be a
                    suffix composed of a train/test label and numerical value, such
                    as "-srcxyz-abc". The numerical id (xyz) should typically
                    be a monotonic sequence. For example, it is automatically
                    generated with the batch number in AV.generate_dataset_activations.
                    Additionally, (abc) could be a unique identifying number. For
                    example, it is automatically created in AV.generate_activations
                    from datetime. Assumes identifier is same for all layers if a
                    list of `layers` is provided.
        """
        if isinstance(layers, str):
            layers = [layers]
        if isinstance(act_tensors, Tensor):
            act_tensors = [act_tensors]

        if len(layers) != len(act_tensors):
            raise ValueError("The dimension of `layers` and `act_tensors` must match!")

        av_dir = AV._assemble_model_dir(path, model_id)

        if not os.path.exists(av_dir):
            os.makedirs(av_dir)

        for i, layer in enumerate(layers):
            av_save_fl_path = AV._assemble_file_path(av_dir, layer, identifier) + ".pt"
            torch.save(act_tensors[i], av_save_fl_path)

    @staticmethod
    def load(
        path: str,
        model_id: str,
        layer: Optional[str] = None,
        identifier: Optional[str] = None,
    ) -> Union[None, DataLoader]:
        r"""
        Loads lazily the activation vectors for given `model_id` and
        `layer` saved under the manifold `path`.

        Args:
            path (str): The manifold path where the activation vectors
                    for the `layer` are stored.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            layer (str or None): The layer for which the activation vectors
                are computed.
            identifier (str or None): An optional identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches. For example, the id could be a
                    suffix composed of a train/test label and numerical value, such
                    as "-srcxyz-abc". The numerical id (xyz) should typically
                    be a monotonic sequence. For example, it is automatically
                    generated with the batch number in AV.generate_dataset_activations.
                    Additionally, (abc) could be a unique identifying number. For
                    example, it is automatically created in AV.generate_activations
                    from datetime.

        Returns:
            dataloader (DataLoader): A torch dataloader that allows to iterate
                    over the activation vectors for given layer.
        """

        av_save_dir = AV._assemble_model_dir(path, model_id)

        if os.path.exists(av_save_dir):
            avdataset = AV.AVDataset(path, model_id, layer, identifier)
            loader = DataLoader(avdataset)
            return loader
        return None

    @staticmethod
    def _manage_loading_layers(
        path: str,
        model_id: str,
        layers: Union[str, List[str]],
        identifier: Optional[str] = None,
        load_from_disk: bool = True,
    ) -> List[str]:
        r"""
        Returns unsaved layers, and deletes saved layers if load_from_disk is False.

        Args:
            path (str): The manifold path where the activation vectors
                    for the `layer` are stored.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            layers (str or List of str): The layer(s) for which the activation vectors
                    are computed.
            identifier (str or None): An optional identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches. For example, the id could be a
                    suffix composed of a train/test label and numerical value, such
                    as "-srcxyz-abc". The numerical id (xyz) should typically
                    be a monotonic sequence. For example, it is automatically
                    generated with the batch number in AV.generate_dataset_activations.
                    Additionally, (abc) could be a unique identifying number. For
                    example, it is automatically created in AV.generate_activations
                    from datetime.

        Returns:
            List of layer names for which activations should be generated
        """

        layers = [layers] if isinstance(layers, str) else layers
        unsaved_layers = []

        if load_from_disk:
            for layer in layers:
                if not AV.exists(path, model_id, layer, identifier):
                    unsaved_layers.append(layer)
        else:
            unsaved_layers = layers
            warnings.warn(
                "Overwriting activations: load_from_disk is set to False. Removing all "
                f"activations matching specified parameters {{path: {path}, "
                f"model_id: {model_id}, layers: {layers}, identifier: {identifier}}} "
                "before generating new activations."
            )
            for layer in layers:
                files = glob.glob(
                    AV._construct_file_search(path, model_id, layer, identifier)
                )
                for filename in files:
                    os.remove(filename)

        return unsaved_layers

    @staticmethod
    def generate_activation(
        path: str,
        module: Module,
        model_id: str,
        layers: Union[str, List[str]],
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        identifier: Optional[str] = None,
        additional_forward_args: Any = None,
        load_from_disk: bool = True,
    ) -> List[Union[Tensor, Tuple[Tensor, ...]]]:
        r"""
        Computes layer activations for the given inputs and specified `layers`

        Args:
            path (str): The manifold path where the activation vectors
                    for the `layer` are stored.
            module (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            layers (str or List of str): The layer(s) for which the activation vectors
                    are computed.
            inputs (tensor or tuple of tensors): Batch of examples for which influential
                    instances are computed. They are passed to the forward_func. The
                    first dimension in `inputs` tensor or tuple of tensors corresponds
                    to the batch size.
            identifier (str or None): An optional identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches. For example, the id could be a
                    suffix composed of a train/test label and numerical value, such
                    as "-srcxyz-abc". The numerical id (xyz) should typically
                    be a monotonic sequence. For example, it is automatically
                    generated with the batch number in AV.generate_dataset_activations.
                    Additionally, (abc) could be a unique identifying number. For
                    example, it is automatically created in AV.generate_activations
                    from datetime.
            additional_forward_args (optional):  Additional arguments that will be
                    passed to forward_func after inputs.
            load_from_disk (bool): Forces function to regenerate activations if False.

        Returns:
            List of Pytorch Tensor or Tuple of Tensors:
                    Activations of each neuron in given layer output. Activations will
                    always be the same size as the output of the given layer.
                    Activations are returned in a tuple if the layer inputs / outputs
                    contain multiple tensors, otherwise a single tensor is returned.
                    Attributions are returned as a list corresponding to the list of
                    layers for which activations were generated.
        """

        unsaved_layers = AV._manage_loading_layers(
            path, model_id, layers, identifier, load_from_disk
        )
        layer_modules = [
            common._get_module_from_name(module, layer) for layer in unsaved_layers
        ]
        if len(unsaved_layers) > 0:
            layer_act = LayerActivation(module, layer_modules)
            new_activations = layer_act.attribute(inputs, additional_forward_args)

            num_id = str(int(datetime.now().microsecond))
            ids = num_id if identifier is None else "-".join([identifier, num_id])
            AV.save(path, model_id, unsaved_layers, new_activations, ids)

        activations = []
        layers = [layers] if isinstance(layers, str) else layers
        for layer in layers:
            if not AV.exists(path, model_id, layer, identifier):
                raise RuntimeError(f"Layer {layer} was not found in manifold")
            else:
                layer_act = []
                for act in AV.load(path, model_id, layer, identifier):
                    layer_act.append(act.squeeze(0))
                layer_act = torch.cat(layer_act)
                activations.append(layer_act)

        return activations

    @staticmethod
    def _unpack_data(data: Union[Any, Tuple[Any, Any]]) -> Any:
        r"""
        Helper to extract input from labels when getting items from a Dataset. Assumes
        that data is either a single value, or a tuple containing two elements.
        The input could itself be a Tuple containing multiple values. If your
        dataset returns a Tuple with more than 2 elements, please reformat it such that
        all inputs are formatted into a tuple stored at the first position.
        """
        if isinstance(data, tuple) or isinstance(data, list):
            data = data[0]
        return data

    r"""TODO:
    1. Can propagate saving labels along with activations.
    2. Use of additional_forward_args when sourcing from dataset?
    """

    @staticmethod
    def generate_dataset_activations(
        path: str,
        module: Module,
        model_id: str,
        layers: Union[str, List[str]],
        src_dataset: Dataset,
        batch_size: int = 1,
        identifier: str = None,
        load_from_disk: bool = True,
    ) -> None:
        r"""
        Computes layer activations for a source dataset and specified `layers`. Assumes
        that the dataset returns a single value, or a tuple containing two elements
        (see AV._unpack_data).

        Args:
            path (str): The manifold path where the activation vectors
                    for the `layer` are stored.
            module (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            layers (str or List of str): The layer(s) for which the activation vectors
                    are computed.
            src_dataset (torch.utils.data.Dataset): Dataset for which influential
                    instances are computed. They are passed to the forward_func.
            batch_size (int): The batch size to compute activations.
            identifier (str or None): An optional identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches. For example, the id could be a
                    suffix composed of a train/test label and numerical value, such
                    as "-srcxyz-abc". The numerical id (xyz) should typically
                    be a monotonic sequence. For example, it is automatically
                    generated with the batch number in AV.generate_dataset_activations.
                    Additionally, (abc) could be a unique identifying number. For
                    example, it is automatically created in AV.generate_activations
                    from datetime.
            load_from_disk (bool): Forces function to regenerate activations if False.

        Returns:
            None. All generate activations are saved on the filesystem.
        """

        unsaved_layers = AV._manage_loading_layers(
            path, model_id, layers, identifier, load_from_disk
        )

        if identifier is None:
            identifier = ""

        if len(unsaved_layers) > 0:
            src_dataloader = DataLoader(src_dataset, batch_size, shuffle=False)
            for i, data in enumerate(src_dataloader):
                AV.generate_activation(
                    path,
                    module,
                    model_id,
                    layers,
                    AV._unpack_data(data),
                    identifier + str(i),
                )

    @staticmethod
    def sort_files(files: List[str]) -> List[str]:
        r"""
        Utility for sorting files based on natural sorting instead of the default
        lexigraphical sort.
        """

        def split_alphanum(s):
            r"""
            Splits string into a list of strings and numbers
                "z23a" -> ["z", 23, "a"]
            """

            return [int(x) if x.isdigit() else x for x in re.split("([0-9]+)", s)]

        return sorted(files, key=split_alphanum)

    class AVDataset(Dataset):
        r"""
        This dataset enables access to activation vectors for a given `model` stored
        under a pre-defined path in the manifold.
        The iterator of this dataset returns a batch of data tensors.
        Additionally, subsets of the model activations can be loaded based on layer
        or identifier .
        """

        def __init__(
            self,
            path: str,
            model_id: str,
            layer: Optional[str] = None,
            identifier: Optional[str] = None,
        ):
            r"""
            Loads into memory the list of all activation file paths associated
            with the input `model_id`.

            Args:
                path (str): The manifold path where the activation vectors
                        for the `layer` are stored.
                model_id (str): The name/version of the model for which layer
                        activations are being computed and stored.
                layer (str or None): The layer for which the activation vectors
                        are computed.
                identifier (str or None): An optional identifier for the layer
                        activations. Can be used to distinguish between activations for
                        different training batches. For example, the id could be a
                        suffix composed of a train/test label and numerical value, such
                        as "-srcxyz-abc". The numerical id (xyz) should typically
                        be a monotonic sequence. For example, it is automatically
                        generated with the batch number in
                        AV.generate_dataset_activations. Additionally, (abc) could be a
                        unique identifying number. For example, it is automatically
                        created in AV.generate_activations from datetime.
            """

            self.av_filesearch = AV._construct_file_search(
                path, model_id, layer, identifier
            )

            files = glob.glob(self.av_filesearch)
            self.files = AV.sort_files(files)

        def __getitem__(self, idx: int) -> Tensor:
            assert idx < len(self.files), "Layer index is out of bounds!"
            fl = self.files[idx]
            av = torch.load(fl)
            return av

        def __len__(self):
            return len(self.files)

