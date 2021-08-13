import base64
import os.path
import pickle as pkl
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from captum.compare.fb._core.CKASimilarity import CKASimilarity
from captum.compare.fb._utils.activations import layer_attributions
from captum.compare.fb._utils.cka import gram_rbf
from captum.compare.fb._utils.similarity import _permutate_lists
from captum.compare.fb._utils.transform import avg_pool_4d
from captum.compare.fb.ui.backend.cluster_gram import reorder_matrix, save_matrix
from flask import Flask, send_from_directory
from flask_cors import CORS
from torch import Tensor, nn

r"""
Initial API setup. Currently, hardcoded for use with pretrained Resnet18,
Resnet50, and CIFAR-10. In the future, this will be completely generic.
"""
# Specify layers for comparisons
model1_layers = ["conv1", "layer1.0.conv1", "layer1.0.conv2", "layer2.0.conv1"]
model2_layers = ["conv1", "layer1.0.conv1", "layer1.0.conv2", "layer2.0.conv1"]

# Initialize dataset
torch.cuda.empty_cache()
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
cifar_trainset = datasets.CIFAR10(
    root="../dotsync-home/data", train=True, download=True, transform=transform
)

dataset_dirname = "cifar10.dataset.log"
img_dataset = datasets.CIFAR10(root=dataset_dirname, train=True, download=True)
stats = {}
# We save comparison results to disk to save computation time
class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_gram_matrices(
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


def generate_cluster_img_bytes(gram: Tensor):
    io = BytesIO()
    clustered_gram, indexes = reorder_matrix(gram.cpu().numpy())
    save_matrix(clustered_gram, "Gram Matrix", io)
    io.seek(0)
    return base64.b64encode(io.getvalue()).decode("utf-8"), indexes


def generate_cluster_img_bytes_all(
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
        clustered_gram, indexes = generate_cluster_img_bytes(gram)
        pkl.dump((clustered_gram, indexes), open(file_name, "wb"))
        clustered_gram_dict[layer] = (clustered_gram, indexes)
    return clustered_gram_dict


model1 = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
).cuda()
model2 = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True
).cuda()

trainloader = torch.utils.data.DataLoader(
    cifar_trainset, batch_size=5000, shuffle=False, num_workers=1
)

trainiter = iter(trainloader)
inputs, labels = next(trainiter)

inputs = inputs.cuda()

model_pairs = _permutate_lists(model1_layers, model2_layers)

cka_sim = CKASimilarity(model1, model2, model1_id="resnet20", model2_id="resnet32")



# model2_clustered_gram_dict = generate_cluster_img_bytes_all(model2_layers, model2_gram_dict, save_dict=".", model_id='resnet32')

gram_dict = {"resnet20": model1_gram_dict, "resnet32": model2_gram_dict}

clustered_gram_dict = {
    "resnet20": model1_clustered_gram_dict,
    "resnet32": model2_clustered_gram_dict,
}

cka_scores = []
for layer1 in model1_layers:
    row = []
    for layer2 in model2_layers:
        row.append(stats[layer1][layer2]["similarity"].item())
    cka_scores.append(row)

this_filepath = Path(os.path.abspath(__file__))
this_dirpath = this_filepath.parent.parent
app = Flask(__name__, static_folder=str(this_dirpath.joinpath("frontend", "build")))
CORS(app)  # TODO: remove this after frontend integration


def _serve(subpath="index.html"):
    return send_from_directory(app.static_folder, subpath)


app.add_url_rule("/", view_func=_serve)
app.add_url_rule("/<path:subpath>", view_func=_serve)


@app.route("/image/<id>")
def retrieve_image(id):
    pil_img = img_dataset[int(id)][0]
    img_io = BytesIO()
    pil_img.save(img_io, format="PNG")
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode("utf-8")


@app.route("/info")
def get_info():
    r"""
    Endpoint for returning general model information. This is
    currently set to resnet18 and resnet50.
    """
    return {
        "model1": {"name": "resnet20", "layers": model1_layers},
        "model2": {"name": "resnet32", "layers": model2_layers},
    }

@app.route("/similarity-grid")
def similarityGrid():
    return {"success": True, "data": stats}

# @app.route("/")

@app.route("/compare/<layer1>/<layer2>")
def compare(layer1, layer2):
    if not (layer1 in stats and layer2 in stats[layer1]):
        return {"success": False, "error": "Layer comparison does not exist"}
    return stats[layer1][layer2]["similarity"]

@app.route("/clustered-gram/<model>/<layer>")
def gram(model, layer):
    if model in gram_dict:
        return clustered_gram_dict[model][layer][0]
    else:
        return {"success": False, "error": "model not found"}

@app.route("/cluster-images/<model>/<layer>/<startIdx>/<endIdx>")
def cluster_images(model, layer, startIdx, endIdx):
    if model in gram_dict:
        startIdx = int(startIdx)
        endIdx = int(endIdx)
        startSlice = startIdx
        endSlice = endIdx
        if startSlice > endSlice:
            endSlice = startIdx
            startSlice = endIdx
        _, indexes = clustered_gram_dict[model][layer]
        image_indexes = indexes[startSlice:endSlice]
        return {"success": True, "data": image_indexes}
    else:
        return {"success": False, "error": "model not found"}

@app.route("/similarities")
def similarities():
    return {"success": True, "data": cka_scores}


if __name__ == "__main__":
    app.run(debug=True)
