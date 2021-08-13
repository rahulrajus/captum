from captum.compare.fb._core.SVCCASimilarity import SVCCASimilarity
from captum.compare.fb._core.CCASimilarity import CCASimilarity
from captum.compare.fb._core.CKASimilarity import CKASimilarity
from tests.helpers.basic_models import BasicModel_ConvNet, BasicModelWithReusableModules
from captum.compare.fb._utils.transform import avg_pool_4d, flatten
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

torch.cuda.empty_cache()



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_trainset = datasets.CIFAR10(root='../dotsync-home/data', train=True, download=True, transform=transform)

model1 = models.resnet18(pretrained=True)
model2 = models.resnet50(pretrained=True)
print(model1)
trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=5000,
                                         shuffle=False, num_workers=1)
trainiter = iter(trainloader)
inputs, labels = trainiter.next()
print("done")

# print(model1)
cca_sim = SVCCASimilarity(model1, model2, model1_id="resnet18", model2_id="resnet50")
out, stats = cca_sim.compare(
            inputs=inputs,
            layer_pairs=("conv1", "conv1"),
            input_identifier='cifar',
            transform=None,
            batch_size=1000
        )
coef_x = stats['conv1']['conv1']['similarity']
print(coef_x)
dir1 = stats['conv1']['conv1']['cca_dirns1'][1, :]
dir2 = stats['conv1']['conv1']['cca_dirns2'][1, :]

dir11 = stats['conv1']['conv1']['cca_dirns1'][1300, :]
dir22 = stats['conv1']['conv1']['cca_dirns2'][1300, :]

coefs = stats['conv1']['conv1']["cca_coef"]
print(coefs[0]/torch.sum(coefs[1:]))
print(dir1.T.shape)
plt.scatter(dir1.T, dir2.T)
plt.scatter(dir11.T, dir22.T, c='red')
plt.show()
plt.savefig('cca_dir.png')

# print(coef_x)
# print(coef_x.shape)
