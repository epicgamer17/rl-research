import unittest
import torch
import numpy as np
import os
import shutil
from stats.latent_tsne import LatentTSNEVisualizer


class TestLatentTSNE(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_output_tsne"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_vectors(self):
        # Test with simple 1D latent vectors (N, D)
        # Keep N > perplexity (default 30), or lower perplexity
        N, D = 50, 32
        latents = torch.randn(N, D)

        # Lower perplexity for small dataset
        visualizer = LatentTSNEVisualizer(n_components=2, perplexity=5.0, n_iter=250)
        points = visualizer.fit_transform(latents)

        self.assertEqual(points.shape, (N, 2))
        self.assertTrue(isinstance(points, np.ndarray))

    def test_images(self):
        # Test with "image" latents (N, C, H, W)
        N, C, H, W = 20, 3, 16, 16
        latents = torch.randn(N, C, H, W)

        visualizer = LatentTSNEVisualizer(n_components=2, perplexity=5.0, n_iter=250)
        points = visualizer.fit_transform(latents)

        self.assertEqual(points.shape, (N, 2))

    def test_plotting(self):
        # Test plotting functionality (just ensure no crash and file creation)
        N, D = 20, 16
        latents = torch.randn(N, D)
        labels = np.random.randint(0, 3, size=N)

        visualizer = LatentTSNEVisualizer(n_components=2, perplexity=5.0, n_iter=250)
        save_path = os.path.join(self.output_dir, "test_plot_tsne.png")

        # Should not raise error
        visualizer.plot(latents, labels=labels, save_path=save_path, show=False)

        self.assertTrue(os.path.exists(save_path))


if __name__ == "__main__":
    unittest.main()
