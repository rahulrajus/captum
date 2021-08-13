import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from captum.compare.fb._core.CKASimilarity import CKASimilarity
from captum.compare.fb._utils.similarity import _permutate_lists
from captum.compare.fb.ui.backend.cka_explorer import CKASimilarityExplorer

model1 = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
).cuda()
model2 = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True
).cuda()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
cifar_trainset = datasets.CIFAR10(
    root="../dotsync-home/data", train=True, download=True, transform=transform
)
dataset_dirname = "cifar10.dataset.log"

img_dataset = datasets.CIFAR10(root=dataset_dirname, train=True, download=True)


trainloader = torch.utils.data.DataLoader(
    cifar_trainset, batch_size=5000, shuffle=False, num_workers=1
)

trainiter = iter(trainloader)
inputs, labels = next(trainiter)

inputs = inputs.cuda()

model1_layers = ["conv1", "layer1.0.conv1", "layer1.0.conv2", "layer2.0.conv1"]
model2_layers = ["conv1", "layer1.0.conv1", "layer1.0.conv2", "layer2.0.conv1"]
model_pairs = _permutate_lists(model1_layers, model2_layers)

cka_sim = CKASimilarity(model1, model2, model1_id="resnet20", model2_id="resnet32")

cka_similarity_explorer = CKASimilarityExplorer(service=cka_sim,
                                                inputs=inputs,
                                                dataset=img_dataset,
                                                save_dir=".",
                                                model1_id="resnet20",
                                                model2_id="resnet32",
                                                model1_layers=model1_layers,
                                                model2_layers=model2_layers,
                                                input_identifier='cifar')
cka_similarity_explorer.start()
