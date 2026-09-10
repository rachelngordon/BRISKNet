import unittest

import numpy as np
import torch

from utils import _ktraj_to_sigpy_coord


class GraspTrajectoryConversionTest(unittest.TestCase):
    def test_converts_radians_to_grid_units(self):
        ktraj = torch.tensor(
            [
                [[-np.pi], [0.0], [np.pi]],
                [[-np.pi / 2], [0.0], [np.pi / 2]],
            ],
            dtype=torch.float32,
        )

        coord = _ktraj_to_sigpy_coord(
            ktraj,
            samples_per_spoke=3,
            image_shape=(320, 192),
        )

        self.assertEqual(coord.shape, (1, 1, 3, 2))
        np.testing.assert_allclose(coord[0, 0, :, 0], [-160.0, 0.0, 160.0])
        np.testing.assert_allclose(coord[0, 0, :, 1], [-48.0, 0.0, 48.0])

    def test_rejects_invalid_image_shape(self):
        ktraj = torch.zeros((2, 4, 1), dtype=torch.float32)

        for image_shape in ((0, 320), (320, -1), (320,)):
            with self.subTest(image_shape=image_shape):
                with self.assertRaisesRegex(ValueError, "positive 2D image shape"):
                    _ktraj_to_sigpy_coord(
                        ktraj,
                        samples_per_spoke=4,
                        image_shape=image_shape,
                    )


if __name__ == "__main__":
    unittest.main()
