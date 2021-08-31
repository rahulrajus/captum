# Representation Similarity
## CKA and CCA UI Explorers

<img width="1680" alt="cka_deep_layer_cifar" src="https://user-images.githubusercontent.com/9889324/131476431-e6a9684e-66ef-4a8d-8cfe-34e84b1a8a0a.png">
<img width="1680" alt="cca_early_layer_cifar" src="https://user-images.githubusercontent.com/9889324/131476466-35083b2b-773b-43ad-b131-249e7ac3332b.png">

Over successive layers, deep neural networks transform the data space so that its dimensions fit the task. Applying representation similarity algorithms across intermediate data spaces sheds light into the major features captured by the layers, and can visually surface potential deficiencies of the models or of the training data. This branch contains 2 UIs: CKA and CCA Explorer. The first UI looks under the hood of the CKA algorithm by using the element wise product of the Gramian matrices for each layer activation in the comparison. The second UI is based on the CCA class of algorithm and projects the input dataset on the CCA dimensionss, allowing users to see clear separations of image/object types on either ends of each dimenison


### CKA UI

### Example notebook
[Run on Colab](https://colab.research.google.com/drive/1cGK7VxPuZQIQ3oK5zROAPuRQlZvFc4k7?usp=sharing)

### CKA UI

### Example notebook
[Run on Colab](https://colab.research.google.com/drive/1u80sBpwocgeKVILCRV2kXV_EopQE5KpD?usp=sharing)

## Thank You

If you have any suggestions or feedback, we will be happy to hear from you!
