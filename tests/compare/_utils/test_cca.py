import random

import numpy as np
import torch

from captum.compare.fb._utils.cca import (
    cov,
    positivedef_matrix_sqrt,
    subarray_by_sum,
    get_cca_similarity,
)
from tests.helpers.basic import BaseTest, assertTensorAlmostEqual


class Test(BaseTest):
    def test_cov(self) -> None:
        x = np.arange(21.0).reshape(7, 3)
        x_torch = torch.from_numpy(x)
        expected_cov = np.cov(x)
        my_cov = cov(x_torch)
        assertTensorAlmostEqual(self, my_cov, expected_cov)

    def test_subarray_by_sum(self) -> None:
        x = torch.tensor([0.5, 0.3, 0.1, 0.1])
        threshold = 0.9
        expected_idx = 2
        actual_idx = subarray_by_sum(x, threshold)
        self.assertEqual(actual_idx, expected_idx)

        # case where threshold never met!
        x = torch.tensor([0.5, 0.3, 0.1, 0.05])
        threshold = 1.0
        expected_idx = 3
        actual_idx = subarray_by_sum(x, threshold)
        self.assertEqual(actual_idx, expected_idx)

    def test_positivedef_matrix_sqrt(self) -> None:
        positivedef = torch.tensor(
            [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
        )

        # use sqrt result from cca author's code
        expected_sqrt = np.array(
            [
                [1.36038826, -0.38268343, -0.0538253],
                [-0.38268343, 1.30656296, -0.38268343],
                [-0.0538253, -0.38268343, 1.36038826],
            ]
        )

        actual_sqrt = positivedef_matrix_sqrt(positivedef)

        assertTensorAlmostEqual(self, actual_sqrt, expected_sqrt)

    def test_cca(self) -> None:
        # Basic Tests
        device = "cpu"
        act = np.array(
            [
                [
                    -7.0318729e-01,
                    -4.9028236e-01,
                    -3.2181433e-01,
                    -1.7550787e00,
                    2.0666447e-01,
                    -2.0112646e00,
                    -5.5725068e-01,
                    3.3721700e-01,
                ],
                [
                    1.5488360e00,
                    -1.3707366e00,
                    1.4252914e00,
                    -2.7946392e-01,
                    -5.5962789e-01,
                    1.1863834e00,
                    1.6985189e00,
                    -1.6912202e00,
                ],
                [
                    -6.9952285e-01,
                    5.8296287e-01,
                    9.7822261e-01,
                    -1.2173721e00,
                    -1.3293954e00,
                    -1.4547423e-03,
                    -1.3146527e00,
                    -3.7961173e-01,
                ],
                [
                    1.2652106e00,
                    1.2066774e-01,
                    1.4794178e-01,
                    -2.7537258e00,
                    -3.5689631e-01,
                    7.7178366e-03,
                    1.4782772e00,
                    -9.5761460e-01,
                ],
            ],
            dtype=np.float32,
        )
        acts1 = torch.from_numpy(act[:2]).to(device)
        acts2 = torch.from_numpy(act[2:]).to(device)

        actual_stats = get_cca_similarity(acts1, acts2, compute_directions=True)
        print(actual_stats.keys())
        expected_stats = {
            "coef_x": torch.tensor(
                [[-0.53679651, -0.84371175], [-0.84371175, 0.53679651]]
            ),
            "invsqrt_xx": torch.tensor(
                [[1.76416132, 0.26268696], [0.26268696, 1.0501397]]
            ),
            "full_coef_x": torch.tensor(
                [[-0.53679651, -0.84371175], [-0.84371175, 0.53679651]]
            ),
            "full_invsqrt_xx": torch.tensor(
                [[1.76416132, 0.26268696], [0.26268696, 1.0501397]]
            ),
            "coef_y": torch.tensor(
                [[0.0875585, -0.99615938], [-0.99615938, -0.0875585]]
            ),
            "invsqrt_yy": torch.tensor(
                [[1.51123762, -0.08379812], [-0.08379812, 1.00618278]]
            ),
            "full_coef_y": torch.tensor(
                [[0.0875585, -0.99615938], [-0.99615938, -0.0875585]]
            ),
            "full_invsqrt_yy": torch.tensor(
                [[1.51123762, -0.08379812], [-0.08379812, 1.00618278]]
            ),
            "cca_coef": torch.tensor([0.77141192, 0.02611916]),
            "x_idxs": torch.tensor([True, True]),
            "y_idxs": torch.tensor([True, True]),
            "mean": torch.tensor(0.7714),
            "cca_directions1": torch.tensor(
                [
                    [
                        -1.95292619,
                        0.7967401,
                        -2.2717259,
                        1.15405227,
                        -0.85075985,
                        -0.0520231,
                        -2.27719981,
                        0.15884596,
                    ],
                    [
                        0.7465154,
                        -0.53908518,
                        0.19037838,
                        1.53844639,
                        -1.20071173,
                        2.38507591,
                        0.60107886,
                        -1.76371691,
                    ],
                ]
            ),
            "cca_directions2": torch.tensor(
                [
                    [
                        -1.89210692,
                        -0.4597549,
                        -0.40199602,
                        2.05388413,
                        -0.39026269,
                        -0.47183052,
                        -2.23997451,
                        0.42121734,
                    ],
                    [
                        0.27734417,
                        -1.63865102,
                        -2.23091435,
                        1.07171403,
                        1.2284539,
                        -0.76261497,
                        1.19788277,
                        -0.1916361,
                    ],
                ]
            ),
        }

        for stat in expected_stats:
            expected_tensor = expected_stats[stat]
            actual_tensor = actual_stats[stat]
            if actual_tensor.dtype is torch.float32:
                assertTensorAlmostEqual(self, actual_tensor, expected_tensor)
            else:
                self.assertTrue(
                    torch.equal(expected_tensor, actual_tensor),
                    msg=f"{stat} does not match",
                )
