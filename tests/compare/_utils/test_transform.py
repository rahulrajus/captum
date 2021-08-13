import torch

from captum.compare.fb._utils.transform import transform_activation
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class Test(BaseTest):
    def test_transform(self) -> None:
        torch.random.manual_seed(1337)

        def custom_avgpool(act):
            return torch.mean(act, axis=(1, 2))

        def custom_flatten(act):
            return act.reshape((len(act), -1))

        conv_act = torch.randn(100, 10, 2, 2)
        fc_act = torch.randn(100, 10)
        conv_avgpool = custom_avgpool(conv_act)
        fc_flatten = custom_flatten(fc_act)

        transform_conv_func = transform_activation(conv_act, custom_avgpool)

        assertTensorAlmostEqual(self, transform_conv_func, conv_avgpool)

        transform_fc_custom = transform_activation(fc_act, custom_flatten)

        assertTensorAlmostEqual(self, transform_fc_custom, fc_flatten)
