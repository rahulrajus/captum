import base64
import os.path
import pickle as pkl
from io import BytesIO

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from captum.compare.fb._core.SVCCASimilarity import SVCCASimilarity
from captum.compare.fb._utils.similarity import _permutate_lists
from captum.compare.fb._utils.transform import avg_pool_4d
from flask import Flask
from flask_cors import CORS


r"""
Initial API setup. Currently, hardcoded for use with pretrained Resnet18,
Resnet50, and CIFAR-10. In the future, this will be completely generic.
"""
# Specify layers for comparisons
model1_layers = ["conv1", "layer1.0.conv1", "layer1.0.conv2", "layer2.0.conv1", "layer3.2.conv2"]
model2_layers = ["conv1", "layer1.0.conv1", "layer1.0.conv2", "layer2.0.conv1", "layer3.2.conv2"]

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
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
load_file = False
if not os.path.isfile("resnet20.resnet32.cifar10-pretrained.avgpool.stats.log") or not load_file:
    # Define Models
    model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).cuda()
    model2 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).cuda()

    trainloader = torch.utils.data.DataLoader(
        cifar_trainset, batch_size=5000, shuffle=False, num_workers=1
    )
    trainiter = iter(trainloader)
    inputs, labels = next(trainiter)

    inputs = inputs.cuda()

    model_pairs = _permutate_lists(model1_layers, model2_layers)

    cca_sim = SVCCASimilarity(
        model1, model2, model1_id="resnet20", model2_id="resnet32"
    )
    out, stats = cca_sim.compare(
        inputs=inputs,
        layer_pairs=model_pairs,
        input_identifier="cifar",
        transform=avg_pool_4d,
        batch_size=1000,
        compute_directions=True,
    )
    pkl.dump(stats, open("resnet20.resnet32.cifar10-pretrained.avgpool.stats.log", "wb"))
else:
    stats = CPU_Unpickler(open("resnet20.resnet32.cifar10-pretrained.avgpool.stats.log", "rb")).load()

cca_scores = []
for layer1 in model1_layers:
    row = []
    for layer2 in model2_layers:
        row.append(stats[layer1][layer2]["similarity"].item())
    cca_scores.append(row)

num_directions = 5

app = Flask(__name__)
CORS(app)  # TODO: remove this after frontend integration


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


@app.route("/compare/<layer1>/<layer2>")
def compare(layer1, layer2):
    if not (layer1 in stats and layer2 in stats[layer1]):
        return {"success": False, "error": "Layer comparison does not exist"}
    dir1 = stats[layer1][layer2]["cca_directions1"]
    dir2 = stats[layer1][layer2]["cca_directions2"]

    out = {}
    out["similarity"] = stats[layer1][layer2]["similarity"].item()
    out["layer1_dimensions"] = []
    out["layer2_dimensions"] = []
    for i in range(num_directions):
        U_sorted_img = torch.argsort(dir1[i, :].T).cpu().tolist()
        V_sorted_img = torch.argsort(dir2[i, :].T).cpu().tolist()

        out["layer1_dimensions"].append({"ids": U_sorted_img})
        out["layer2_dimensions"].append({"ids": V_sorted_img})
    return {"success": True, "data": out}


@app.route("/similarities")
def similarities():
    return {"success": True, "data": cca_scores}


if __name__ == "__main__":
    app.run(debug=True)
