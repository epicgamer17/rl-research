import unittest
import torch
import numpy as np
import os
import shutil
from stats.latent_umap import LatentUMAPVisualizer


class TestLatentUMAP(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_output_umap"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_vectors(self):
        # Test with simple 1D latent vectors (N, D)
        # UMAP usually likes N > n_neighbors
        N, D = 50, 32
        latents = torch.randn(N, D)

        visualizer = LatentUMAPVisualizer(n_components=2, n_neighbors=10)
        points = visualizer.fit_transform(latents)

        self.assertEqual(points.shape, (N, 2))
        self.assertTrue(isinstance(points, np.ndarray))

        # Test separate transform
        # Create new points
        new_latents = torch.randn(5, D)
        new_points = visualizer.transform(new_latents)
        self.assertEqual(new_points.shape, (5, 2))

    def test_images(self):
        # Test with "image" latents (N, C, H, W)
        N, C, H, W = 20, 3, 16, 16
        latents = torch.randn(N, C, H, W)

        visualizer = LatentUMAPVisualizer(n_components=2, n_neighbors=5)
        points = visualizer.fit_transform(latents)

        self.assertEqual(points.shape, (N, 2))

    def test_plotting(self):
        # Test plotting functionality (just ensure no crash and file creation)
        N, D = 20, 16
        latents = torch.randn(N, D)
        labels = np.random.randint(0, 3, size=N)

        visualizer = LatentUMAPVisualizer(n_components=2, n_neighbors=5)
        save_path = os.path.join(self.output_dir, "test_plot_umap.png")

        # Should not raise error
        visualizer.plot(latents, labels=labels, save_path=save_path, show=False)

        self.assertTrue(os.path.exists(save_path))


if __name__ == "__main__":
    unittest.main()
