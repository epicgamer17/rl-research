import unittest
import torch
import os
import shutil
from stats.stats import StatTracker, PlotType
import numpy as np


class TestStatsViz(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_graphs"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_chance_probs_plotting(self):
        tracker = StatTracker(model_name="test_model")
        tracker._init_key("chance_probs")
        tracker.add_plot_types(
            "chance_probs", PlotType.BAR, bar_threshold=0.01, show_all_bars=True
        )

        # Create data where index 0 is 1.0 and others are 0.0
        data = torch.zeros((10, 5))
        data[:, 0] = 1.0

        # Manually set stats for testing
        tracker.stats["chance_probs"] = data

        # Plot with all bars
        tracker.plot_graphs(dir=self.test_dir)
        # Verify it runs without error

    def test_latent_pca_saving(self):
        # Use PCA as a proxy for verifying save logic since UMAP might not be installed in test env
        tracker = StatTracker(model_name="test_model")

        # Create dummy latent data (batch, dim)
        latents = torch.randn(10, 10)
        labels = torch.randint(0, 2, (10,))

        tracker.add_latent_visualization(
            "latent_viz", latents, labels=labels, method="pca", n_components=2
        )

        print("\nPlotting graphs...")
        tracker.plot_graphs(dir=self.test_dir)

        expected_file = os.path.join(self.test_dir, "test_model_latent_viz_pca.png")
        if os.path.exists(expected_file):
            print(f"File {expected_file} exists.")
        else:
            print(f"File {expected_file} DOES NOT EXIST.")

        self.assertTrue(
            os.path.exists(expected_file),
            "Latent PCA visualization file was not saved.",
        )

    def test_latent_bfloat16_pca_saving(self):
        # Verify BFloat16 support
        tracker = StatTracker(model_name="test_model_bf16")

        # Create BFloat16 latent data
        latents = torch.randn(10, 10, dtype=torch.bfloat16)
        labels = torch.randint(0, 2, (10,))

        tracker.add_latent_visualization(
            "latent_viz_bf16", latents, labels=labels, method="pca", n_components=2
        )

        print("\nPlotting BF16 graphs...")
        tracker.plot_graphs(dir=self.test_dir)

        expected_file = os.path.join(
            self.test_dir, "test_model_bf16_latent_viz_bf16_pca.png"
        )
        self.assertTrue(
            os.path.exists(expected_file),
            "BFloat16 Latent PCA visualization file was not saved.",
        )


if __name__ == "__main__":
    unittest.main()
