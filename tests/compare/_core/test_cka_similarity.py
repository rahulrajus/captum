import tempfile

import torch
from torch import nn

from captum.compare.fb._core.CKASimilarity import CKASimilarity
from captum.compare.fb._utils.activations import layer_attributions
from captum.compare.fb._utils.cka import cka
from captum.compare.fb._utils.transform import avg_pool_4d, flatten
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.helpers.basic_models import BasicModel_ConvNet, BasicModelWithReusableModules


class Test(BaseTest):
    def test_cka_api_basic(self) -> None:
        model1 = BasicModelWithReusableModules()
        model2 = BasicModelWithReusableModules()

        inputs = torch.rand(100, 3)
        if torch.cuda.is_available():
            model1 = model1.cuda()
            model2 = model2.cuda()
            inputs = inputs.cuda()
        cka_sim = CKASimilarity(model1, model2)
        model_layers = ["lin1", "relu", "lin2"]
        model1_layers = [model1.lin1, model1.relu, model1.lin2]
        model2_layers = [model2.lin1, model2.relu, model2.lin2]
        with tempfile.TemporaryDirectory() as tmpdir:
            act1 = layer_attributions(
                model1, model_layers, inputs, save_dir=tmpdir, model_id="model1"
            )
            act2 = layer_attributions(
                model2, model_layers, inputs, save_dir=tmpdir, model_id="model2"
            )
        # Test 1 layer pair
        score_1pair, _ = cka_sim.compare(
            inputs,
            layer_pairs=("lin2", "lin2"),
            debiased=True,
            kernel="linear",
            bandwidth=0.5,
        )
        out_1pair, _ = cka(
            act1["lin2"], act2["lin2"], debiased=True, kernel="linear", bandwidth=0.5
        )

        assertTensorAlmostEqual(self, actual=score_1pair, expected=out_1pair)
        # Test multiple layer pairs
        score_3pair, stats = cka_sim.compare(
            inputs,
            layer_pairs=[("lin2", "lin2"), ("lin1", "lin2"), ("relu", "relu"),],
            debiased=True,
            kernel="linear",
            bandwidth=0.5,
        )

        out_first, _ = cka(
            act1["lin2"], act2["lin2"], debiased=True, kernel="linear", bandwidth=0.5
        )
        out_second, _ = cka(
            act1["lin1"], act2["lin2"], debiased=True, kernel="linear", bandwidth=0.5
        )
        out_third, _ = cka(
            act1["relu"], act2["relu"], debiased=True, kernel="linear", bandwidth=0.5
        )
        expected_score_3pair = torch.tensor([out_first, out_second, out_third])
        assertTensorAlmostEqual(self, actual=score_3pair, expected=expected_score_3pair)

        # Test if no layer pairs passed in
        out_nopairs, _ = cka_sim.compare(
            inputs, debiased=True, kernel="linear", bandwidth=0.5
        )
        sim_mat = []
        for layer1 in model_layers:
            for layer2 in model_layers:
                score, _ = cka(
                    act1[layer1],
                    act2[layer2],
                    debiased=True,
                    kernel="linear",
                    bandwidth=0.5,
                )
                sim_mat.append(score)
        expected_out_nopairs = torch.tensor(sim_mat)
        assertTensorAlmostEqual(self, actual=out_nopairs, expected=expected_out_nopairs)

    def test_cka_api_conv(self) -> None:
        model1 = BasicModel_ConvNet()
        model2 = BasicModel_ConvNet()
        inputs = torch.rand(100, 1, 10, 10)
        if torch.cuda.is_available():
            model1 = model1.cuda()
            model2 = model2.cuda()
            inputs = inputs.cuda()

        cka_sim = CKASimilarity(model1, model2)
        model_layers = ["conv1"]
        with tempfile.TemporaryDirectory() as tmpdir:
            act1 = layer_attributions(
                model1, model_layers, inputs, model_id="model1", save_dir=tmpdir
            )
            act2 = layer_attributions(
                model2, model_layers, inputs, model_id="model2", save_dir=tmpdir
            )
        score_1pair, _ = cka_sim.compare(
            inputs,
            layer_pairs=("conv1", "conv1"),
            transform=avg_pool_4d,
            debiased=True,
            kernel="linear",
            bandwidth=0.5,
        )
        act1_avgpool = torch.mean(act1["conv1"], axis=(2, 3))
        act2_avgpool = torch.mean(act2["conv1"], axis=(2, 3))

        expected_score_1pair, _ = cka(
            act1_avgpool, act2_avgpool, debiased=True, kernel="linear", bandwidth=0.5
        )

        assertTensorAlmostEqual(self, actual=score_1pair, expected=expected_score_1pair)

    def test_cka_api_transforms(self) -> None:
        model1 = BasicModel_ConvNet()
        model2 = BasicModel_ConvNet()
        inputs = torch.rand(100, 1, 10, 10)
        if torch.cuda.is_available():
            model1 = model1.cuda()
            model2 = model2.cuda()
            inputs = inputs.cuda()
        cka_sim = CKASimilarity(model1, model2)

        model_layers = ["conv1", "pool1", "fc2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            act1 = layer_attributions(
                model1, model_layers, inputs, model_id="model1", save_dir=tmpdir
            )
            act2 = layer_attributions(
                model2, model_layers, inputs, model_id="model2", save_dir=tmpdir
            )

        def custom_flatten(act):
            return act.reshape((len(act), -1))

        score_transform_list, _ = cka_sim.compare(
            inputs,
            layer_pairs=[("conv1", "conv1"), ("conv1", "fc2"), ("pool1", "fc2"),],
            transform=[avg_pool_4d, (flatten, None), (flatten, None)],
            debiased=True,
            kernel="linear",
            bandwidth=0.5,
        )

        act1_conv1_avg = torch.mean(act1["conv1"], axis=(2, 3))
        act1_conv1_flatten = custom_flatten(act1["conv1"])
        act2_conv1_avg = torch.mean(act2["conv1"], axis=(2, 3))
        act1_pool1_flatten = custom_flatten(act1["pool1"])

        out_first, _ = cka(
            act1_conv1_avg, act2_conv1_avg, debiased=True, kernel="linear", bandwidth=0.5
        )
        out_second, _ = cka(
            act1_conv1_flatten, act2["fc2"], debiased=True, kernel="linear", bandwidth=0.5
        )
        out_third, _ = cka(
            act1_pool1_flatten, act2["fc2"], debiased=True, kernel="linear", bandwidth=0.5
        )
        expected_score_transform_list = torch.tensor([out_first, out_second, out_third])

        assertTensorAlmostEqual(
            self, actual=score_transform_list, expected=expected_score_transform_list
        )

    def test_standalone_layers(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cka_sim = CKASimilarity()

        layer1 = nn.Conv2d(1, 2, 3, 1).to(device)
        layer2 = nn.Conv2d(1, 2, 3, 1).to(device)
        layer3 = nn.Linear(4, 8).to(device)
        layer4 = nn.Linear(4, 8).to(device)
        layer5 = nn.Linear(4, 8).to(device)

        input1 = torch.rand(100, 1, 10, 10).to(device)
        input2 = torch.rand(100, 4).to(device)
        input3 = torch.rand(100, 4).to(device)

        layer_pairs = [(layer1, layer2), (layer3, layer4), (layer3, layer5)]

        inputs = [input1, input2, input3]

        out, _ = cka_sim.compare(
            inputs=inputs,
            layer_pairs=layer_pairs,
            transform=avg_pool_4d,
            debiased=True,
            kernel="linear",
            bandwidth=0.5,
        )

        layer1_act1 = torch.mean(layer1(input1), axis=(2, 3))
        layer2_act1 = torch.mean(layer2(input1), axis=(2, 3))
        layer3_act2 = layer3(input2)
        layer4_act2 = layer4(input2)
        layer3_act3 = layer3(input3)
        layer5_act3 = layer5(input3)

        out1, _ = cka(
            layer1_act1, layer2_act1, debiased=True, kernel="linear", bandwidth=0.5
        )
        out2, _ = cka(
            layer3_act2, layer4_act2, debiased=True, kernel="linear", bandwidth=0.5
        )
        out3, _ = cka(
            layer3_act3, layer5_act3, debiased=True, kernel="linear", bandwidth=0.5
        )
        expected_out = torch.tensor([out1, out2, out3])

        assertTensorAlmostEqual(self, out, expected_out)
