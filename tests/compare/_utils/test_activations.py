import glob
import tempfile

import torch

from captum._utils.fb.av import AV
from captum.attr import LayerActivation
from captum.compare.fb._utils.activations import layer_attributions
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual
from tests.helpers.basic_models import BasicModel_ConvNet


class Test(BaseTest):
    def test_activations(self) -> None:
        torch.random.manual_seed(1337)

        model = BasicModel_ConvNet()
        save_model_layers = ["conv1", "pool1", "fc1"]
        mixed_model_layers = ["conv1", "pool1", "fc1", "fc2"]
        inputs = torch.rand(5000, 1, 10, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_activations = layer_attributions(
                model, save_model_layers, inputs, save_dir=tmpdir
            )

            # Test if activations are correct
            layer_act = LayerActivation(
                model, [model.conv1, model.pool1, model.fc1, model.fc2]
            )

            #TODO: switch to hardcoded values
            expected_activations = layer_act.attribute(inputs)

            assertTensorAlmostEqual(self, model_activations['conv1'], expected_activations[0])

            assertTensorAlmostEqual(self, model_activations['pool1'], expected_activations[1])

            assertTensorAlmostEqual(self, model_activations['fc1'], expected_activations[2])

            # Check if all layers exist except for last
            self.assertTrue(AV.exists(tmpdir, "net", "conv1"))
            self.assertTrue(AV.exists(tmpdir, "net", "pool1"))
            self.assertTrue(AV.exists(tmpdir, "net", "fc1"))
            self.assertFalse(AV.exists(tmpdir, "net", "fc2"))

            # add in last layer
            model_activations = layer_attributions(
                model, mixed_model_layers, inputs, save_dir=tmpdir
            )

            self.assertTrue(len(mixed_model_layers) == len(model_activations))
            assertTensorAlmostEqual(self, model_activations['fc2'], expected_activations[3])

            # check if fc2 layer was saved
            self.assertTrue(AV.exists(tmpdir, "net", "fc2"))

            # check if function saves a 5000 example input into batches
            model_path = AV._assemble_model_dir(tmpdir, "net")
            self.assertEqual(len(glob.glob(model_path + "conv1*")), 5)

            # check if loaded activation has 5000 examples
            self.assertEqual(model_activations['conv1'].shape[0], 5000)

            #TODO: check if function is loading existing activations from disk
