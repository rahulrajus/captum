import numpy as np
import torch

from captum.compare.fb._utils.cka import cka, rbf_kernel
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class Test(BaseTest):
    def test_cka_implementation(self) -> None:
        np.random.seed(1337)
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 10) + X

        r"""Following values are obtained from running the CKA Implementation provided by Google in the following notebook:
        https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
        """
        cka_linear_debiased = torch.tensor(0.5134564895055259)
        cka_linear = torch.tensor(0.5576132566117589)
        cka_rbf = torch.tensor(0.5758322230135366)
        cka_rbf_debiased = torch.tensor(0.5016102703531156)
        cka_feature_debiased = torch.tensor(0.513456489505526)
        cka_feature = torch.tensor(0.5576132566117589)

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        torch_cka_linear_debiased = cka(X, Y, kernel="linear", debiased=True)[0]
        torch_cka_linear = cka(X, Y, kernel="linear")[0]
        torch_cka_rbf = cka(X, Y, kernel="rbf")[0]
        torch_cka_rbf_debiased = cka(X, Y, kernel="rbf", debiased=True)[0]
        torch_cka_feature = cka(X, Y, use_features=True, debiased=False)[0]
        torch_cka_feature_debiased = cka(X, Y, use_features=True, debiased=True)[0]
        torch_cka_custom = cka(X, Y, kernel=rbf_kernel)[0]
        torch_cka_custom_debiased = cka(X, Y, kernel=rbf_kernel, debiased=True)[0]

        assertTensorAlmostEqual(self, torch_cka_linear_debiased, cka_linear_debiased)
        assertTensorAlmostEqual(self, torch_cka_linear, cka_linear)
        assertTensorAlmostEqual(self, torch_cka_rbf, cka_rbf)
        assertTensorAlmostEqual(self, torch_cka_rbf_debiased, cka_rbf_debiased)
        assertTensorAlmostEqual(self, torch_cka_feature, cka_feature)
        assertTensorAlmostEqual(self, torch_cka_feature_debiased, cka_feature_debiased)
        assertTensorAlmostEqual(self, torch_cka_custom, cka_rbf)
        assertTensorAlmostEqual(self, torch_cka_custom_debiased, cka_rbf_debiased)
