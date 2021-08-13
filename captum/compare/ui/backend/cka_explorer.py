import base64
import os.path
import pickle as pkl
from io import BytesIO
from typing import Callable, Dict, List

import torch

from captum.compare.fb._utils.activations import layer_attributions
from captum.compare.fb._utils.cka import rbf_kernel
from captum.compare.fb._utils.similarity import _permutate_lists
from captum.compare.fb._utils.transform import avg_pool_4d
from captum.compare.fb.ui.backend.cluster_gram import reorder_matrix, save_matrix
from captum.compare.fb._core.ModuleSimilarity import ModuleSimilarity
from captum.compare.fb.ui.backend.similarity_explorer import SimilarityExplorer
from torch import Tensor, nn

class CKASimilarityExplorer(SimilarityExplorer):
    def __init__(self,
        service: ModuleSimilarity,
        inputs: Tensor,
        dataset,
        save_dir: str,
        model1_id: str,
        model2_id: str,
        model1_layers: List[str],
        model2_layers: List[str],
        input_identifier: str,
        gram_func: Callable = rbf_kernel,
        gram_type: str = "rbf",
        load_gram: bool = True,
        transform: Callable = avg_pool_4d,
        transform_type: str = "avg_pool",
        sigma: float = 1.0,
        debiased: bool = False
    ):
        super().__init__(service)
        self.dataset = dataset
        self.save_dir = save_dir
        self.inputs = inputs
        self.model1_id = model1_id
        self.model2_id = model2_id
        self.model1_layers = model1_layers
        self.model2_layers = model2_layers
        self.input_identifier = input_identifier
        self.gram_func = gram_func
        self.gram_type = gram_type
        self.load_gram = load_gram
        self.transform = transform
        self.transform_type = transform_type
        self.sigma = sigma
        self.debiased = debiased

        self.app.add_url_rule("/image/<id>", view_func=self.retrieve_image)
        self.app.add_url_rule("/info", view_func=self.get_info)
        self.app.add_url_rule("/similarity-grid", view_func=self.similarityGrid)
        self.app.add_url_rule("/compare/<layer1>/<layer2>", view_func=self.compare)
        self.app.add_url_rule("/clustered-gram/<model>/<layer>", view_func=self.gram)
        self.app.add_url_rule("/cluster-images/<model>/<layer>/<startIdx>/<endIdx>", view_func=self.cluster_images)
        self.app.add_url_rule("/similarities", view_func=self.similarities)
        self.generate_comparison()

    def load_gram_matrices(
        self,
        model: nn.Module,
        layers: List,
        inputs: Tensor,
        save_dir: str,
        model_id: str,
        input_identifier: str,
        gram_func: Callable,
        gram_type: str,
        load_gram: bool = True,
        transform: Callable = avg_pool_4d,
        transform_type: str = "avg_pool",
    ):
        gram_dict = {}
        unsaved_layers = []
        for layer in layers:
            file_name = f"{save_dir}/{model_id}.{layer}.{input_identifier}.{gram_type}.{transform_type}.log"
            if os.path.isfile(file_name):
                print("Loading saved gram for: ", model_id, layer)
                gram_dict[layer] = torch.load(file_name)
            else:
                unsaved_layers.append(layer)

        layer_activation_dict = layer_attributions(
            model,
            unsaved_layers,
            inputs,
            save_dir=".",
            model_id=model_id,
            identifier=input_identifier,
        )

        for layer in unsaved_layers:
            act = layer_activation_dict[layer]
            gram = gram_func(transform(act))
            gram_dict[layer] = gram
            torch.save(
                gram,
                f"{save_dir}/{model_id}.{layer}.{input_identifier}.{gram_type}.{transform_type}.log",
            )
        return gram_dict

    def generate_cluster_img_bytes(self, gram: Tensor):
        io = BytesIO()
        clustered_gram, indexes = reorder_matrix(gram.cpu().numpy())
        save_matrix(clustered_gram, "Gram Matrix", io)
        io.seek(0)
        return base64.b64encode(io.getvalue()).decode("utf-8"), indexes


    def generate_cluster_img_bytes_all(
        self,
        layers: List,
        gram_dict: Dict,
        model_id: str,
        input_identifier: str,
        gram_type: str,
        transform_type: str,
        save_dir: str = ".",
    ):
        clustered_gram_dict = {}
        unsaved_layers = []
        for layer in layers:
            file_name = f"{save_dir}/{model_id}.{layer}.{input_identifier}.{gram_type}.{transform_type}.clustered.log"
            if os.path.isfile(file_name):
                print("Loading saved cluster for: ", model_id, layer)
                clustered_gram_dict[layer] = pkl.load(open(file_name, "rb"))
            else:
                unsaved_layers.append(layer)
        for layer in unsaved_layers:
            print("Generating cluster for: ", layer)
            file_name = f"{save_dir}/{model_id}.{layer}.{input_identifier}.{gram_type}.{transform_type}.clustered.log"
            gram = gram_dict[layer]
            clustered_gram, indexes = self.generate_cluster_img_bytes(gram)
            pkl.dump((clustered_gram, indexes), open(file_name, "wb"))
            clustered_gram_dict[layer] = (clustered_gram, indexes)
        return clustered_gram_dict

    def generate_comparison(self):
        layer_pairs = _permutate_lists(self.model1_layers, self.model2_layers)
        out, stats = self.service.compare(
            inputs=self.inputs,
            layer_pairs=layer_pairs,
            input_identifier=self.input_identifier,
            transform=None,
            batch_size=1000,
        )

        model1_gram_dict = self.load_gram_matrices(
            self.service.model1,
            self.model1_layers,
            self.inputs,
            self.save_dir,
            self.model1_id,
            self.input_identifier,
            gram_type=self.gram_type,
            gram_func=self.gram_func,
            transform=self.transform,
            transform_type=self.transform_type,
        )

        model2_gram_dict = self.load_gram_matrices(
            self.service.model2,
            self.model2_layers,
            self.inputs,
            self.save_dir,
            self.model2_id,
            self.input_identifier,
            gram_type=self.gram_type,
            gram_func=self.gram_func,
            transform=self.transform,
            transform_type=self.transform_type,
        )

        model1_clustered_gram_dict = self.generate_cluster_img_bytes_all(
            self.model1_layers,
            model1_gram_dict,
            transform_type=self.transform_type,
            gram_type=self.gram_type,
            input_identifier=self.input_identifier,
            save_dir=self.save_dir,
            model_id=self.model1_id,
        )

        model2_clustered_gram_dict = self.generate_cluster_img_bytes_all(
            self.model2_layers,
            model2_gram_dict,
            transform_type=self.transform_type,
            gram_type=self.gram_type,
            input_identifier=self.input_identifier,
            save_dir=self.save_dir,
            model_id=self.model2_id
        )

        self.stats = stats
        self.gram_dict = {}
        self.gram_dict[self.service.model1_id] = model1_gram_dict
        self.gram_dict[self.service.model2_id] = model2_gram_dict
        self.clustered_gram_dict = {}
        self.clustered_gram_dict[self.service.model1_id] = model1_clustered_gram_dict
        self.clustered_gram_dict[self.service.model2_id] = model2_clustered_gram_dict
        self.cka_scores = []
        for layer1 in self.model1_layers:
            row = []
            for layer2 in self.model2_layers:
                row.append(self.stats[layer1][layer2]["similarity"].item())
            self.cka_scores.append(row)
    ### Endpoints

    def retrieve_image(self, id):
        pil_img = self.dataset[int(id)][0]
        img_io = BytesIO()
        pil_img.save(img_io, format="PNG")
        img_io.seek(0)
        return base64.b64encode(img_io.getvalue()).decode("utf-8")

    def get_info(self):
        r"""
        Endpoint for returning general model information. This is
        currently set to resnet18 and resnet50.
        """
        return {
            "model1": {"name": self.service.model1_id, "layers": self.model1_layers},
            "model2": {"name": self.service.model2_id, "layers": self.model2_layers},
        }

    def similarityGrid(self):
        return {"success": True, "data": self.stats}

    def compare(self, layer1, layer2):
        if not (layer1 in self.stats and layer2 in self.stats[layer1]):
            return {"success": False, "error": "Layer comparison does not exist"}
        return self.stats[layer1][layer2]["similarity"]

    def gram(self, model, layer):
        if model in self.gram_dict:
            return self.clustered_gram_dict[model][layer][0]
        else:
            return {"success": False, "error": "model not found"}

    def cluster_images(self, model, layer, startIdx, endIdx):
        if model in self.gram_dict:
            startIdx = int(startIdx)
            endIdx = int(endIdx)
            startSlice = startIdx
            endSlice = endIdx
            if startSlice > endSlice:
                endSlice = startIdx
                startSlice = endIdx
            _, indexes = self.clustered_gram_dict[model][layer]
            image_indexes = indexes[startSlice:endSlice]
            return {"success": True, "data": image_indexes}
        else:
            return {"success": False, "error": "model not found"}

    def similarities(self):
        return {"success": True, "data": self.cka_scores}
