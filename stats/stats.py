import numpy as np
import torch
import torch.multiprocessing as mp
from queue import Empty
from typing import Dict, Optional, List, Any
import matplotlib.pyplot as plt
from enum import Enum, auto
from stats.latent_pca import LatentPCAVisualizer
from stats.latent_tsne import LatentTSNEVisualizer
try:
    from stats.latent_umap import LatentUMAPVisualizer
except ImportError:
    LatentUMAPVisualizer = None

class PlotType(Enum):
    ROLLING_AVG = auto()
    VARIATION_FILL = auto()
    BEST_FIT_LINE = auto()
    LOG_Y = auto()
    EXPONENTIAL_AVG = auto()
    BAR = auto()


class StatTracker:
    """
    A stat tracker that encapsulates multiprocessing queue logic.
    Can be instantiated as a 'host' (manages data and a queue)
    or a 'client' (sends data to the host's queue).
    """

    def __init__(
        self,
        model_name: str,
        stat_keys: List[str] = None,
        target_values: Optional[Dict[str, float]] = None,
        use_tensor_dicts: Optional[Dict[str, List[str]]] = None,
        # This new argument enables the dual-mode logic
        queue: Optional[mp.Queue] = None,
    ):
        self.model_name = model_name
        self._is_client = queue is not None

        if self._is_client:
            # CLIENT MODE: Only holds a reference to the queue. No data storage.
            self.queue = queue
            self.stats = None  # Clients don't store stats.
        else:
            # HOST MODE: Manages the queue and the actual data.
            self.queue = mp.Queue()
            self.stats = {}
            self.num_steps = 0
            self.time_elapsed = 0.0
            self.targets = target_values or {}
            use_tensor_dicts = use_tensor_dicts or {}
            self.plot_configs = {}
            self.latent_viz_data = {}

            if stat_keys:
                for key in stat_keys:
                    self._init_key(
                        key,
                        target_value=self.targets.get(key),
                        subkeys=use_tensor_dicts.get(key),
                    )

    def get_client(self) -> "StatTracker":
        """Returns a lightweight client instance for passing to a worker process."""
        if self._is_client:
            raise RuntimeError("Cannot get a client from another client.")
        # The client is initialized with the host's queue and no data keys.
        return StatTracker(model_name=self.model_name, queue=self.queue)

    def _init_key(
        self,
        key: str,
        target_value: Optional[float] = None,
        subkeys: Optional[List[str]] = None,
    ):
        """Internal method for the host to initialize a new stat."""
        print(f"Initializing stat '{key}' with subkeys {subkeys}")
        if key in self.stats:
            raise ValueError(f"Stat '{key}' already exists")
        if subkeys:
            self.stats[key] = {subkey: torch.empty(0) for subkey in subkeys}
        else:
            self.stats[key] = torch.empty(0)
        if target_value is not None:
            self.targets[key] = target_value

    def append(self, key: str, value: float, subkey: Optional[str] = None):
        if self._is_client:
            # print("Appending stat {} from client".format(key))
            # Client sends the command to the queue
            self.queue.put(("append", key, value, subkey))
        else:
            # Host executes the command directly
            if key not in self.stats:
                self._init_key(key)
            
            # Prepare the new value as a tensor
            if isinstance(value, torch.Tensor):
                new_val = value.detach().cpu()
            else:
                new_val = torch.tensor([value])

            # Helper to append to a tensor
            def append_to_tensor(current_tensor, new_data):
                if current_tensor.numel() == 0:
                    return new_data
                return torch.cat((current_tensor, new_data))

            if isinstance(self.stats[key], Dict):
                if subkey is None:
                    raise ValueError(f"Stat '{key}' requires a subkey")
                self.stats[key][subkey] = append_to_tensor(self.stats[key][subkey], new_val)
            else:
                self.stats[key] = append_to_tensor(self.stats[key], new_val)

    def increment_steps(self, n: int = 1):
        if self._is_client:
            self.queue.put(("increment_steps", n))
        else:
            self.num_steps += n

    def drain_queue(self):
        """Host-only method to process all messages from clients."""
        if self._is_client:
            raise RuntimeError(
                "drain_queue() can only be called on the host StatTracker."
            )
        while not self.queue.empty():
            try:
                message = self.queue.get_nowait()
                method_name, *args = message
                getattr(self, method_name)(*args)
            except Empty:
                break
            except Exception as e:
                print(f"Error processing stat queue message: {message}, Error: {e}")

    # Other methods (get, plot_graphs, etc.) are host-only and don't need changes.
    # They will correctly fail on a client because `self.stats` is None.
    def get_num_steps(self):
        if self._is_client:
            return None
        return self.num_steps

    def set_time_elapsed(self, time_elapsed: float):
        if self._is_client:
            self.queue.put(("set_time_elapsed", time_elapsed))
        else:
            self.time_elapsed = time_elapsed

    # Other methods like plot_graphs, get, keys, etc. remain unchanged...
    def get_time_elapsed(self):
        if self._is_client:
            return None
        return self.time_elapsed

    def add_plot_types(self, key: str, *plot_types: PlotType, **params: Any):
        """Add extra plot types on top of current ones."""
        if self._is_client:
            return None
        if key not in self.plot_configs:
            self.plot_configs[key] = {"types": set(), "params": {}}
        self.plot_configs[key]["types"].update(plot_types)
        self.plot_configs[key]["params"].update(params)

    def add_latent_visualization(
        self, 
        key: str, 
        latents: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        method: str = 'pca', 
        **kwargs
    ):
        """
        Updates the latent visualization data for a given key.
        Only the latest batch is kept.
        """
        # Detach and move to CPU immediately to avoid holding GPU memory in buffer
        if isinstance(latents, torch.Tensor):
            latents = latents.detach().cpu()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu()

        if self._is_client:
            self.queue.put(("add_latent_visualization", key, latents, labels, method, kwargs))
        else:
            self.latent_viz_data[key] = {
                "latents": latents,
                "labels": labels,
                "method": method,
                "kwargs": kwargs
            }

    def plot_graphs(self, dir: Optional[str] = None):
        if self._is_client:
            raise RuntimeError("Cannot plot graphs from a client instance.")

        collected_stats = {
            key: tensor
            for key, tensor in self.stats.items()
            if (
                (
                    isinstance(tensor, Dict)
                    and any(t.numel() > 0 for t in tensor.values())
                )
                or (isinstance(tensor, torch.Tensor) and tensor.numel() > 0)
            )
        }

        if len(collected_stats) > 0:
            fig, axs = plt.subplots(
                len(collected_stats), 1, figsize=(10, 5 * len(collected_stats))
            )
            if len(collected_stats) == 1:
                axs = [axs]

            for ax, (key, tensor) in zip(axs, collected_stats.items()):
                config = self.plot_configs.get(key, {"types": set(), "params": {}})
                print("plotting {}".format(key))
                if isinstance(tensor, Dict):
                    for subkey, subtensor in tensor.items():
                        print("  subkey {}".format(subkey))
                        self._plot_tensor(ax, subtensor, f"{key}:{subkey}", config)
                else:
                    self._plot_tensor(ax, tensor, key, config)

                # Plot target line if exists as a horizontal line
                if key in self.targets and self.targets[key] is not None:
                    target_value = self.targets[key]
                    ax.axhline(
                        y=target_value,
                        color="r",
                        linestyle="--",
                        label=f"Target: {target_value}",
                    )
                    ax.legend()
            
            plt.tight_layout()
            if dir:
                fig.savefig(f"{dir}/{self.model_name}_stats.png")
            plt.close(fig)
        else:
            fig = None

        # Plot latent visualizations
        for key, data in self.latent_viz_data.items():
            print(f"plotting latent viz {key} using {data['method']}")
            method = data['method'].lower()
            latents = data['latents']
            labels = data['labels']
            kwargs = data['kwargs']
            
            visualizer = None
            if method == 'pca':
                visualizer = LatentPCAVisualizer(**kwargs)
            elif method == 'tsne':
                visualizer = LatentTSNEVisualizer(**kwargs)
            elif method == 'umap':
                if LatentUMAPVisualizer is None:
                    print(f"Skipping UMAP for {key}: umap-learn not installed.")
                    continue
                visualizer = LatentUMAPVisualizer(**kwargs)
            else:
                print(f"Unknown latent visualization method: {method}")
                continue
            
            if visualizer:
                save_path = None
                if dir:
                    save_path = f"{dir}/{self.model_name}_{key}_{method}.png"
                
                # Check dimensionality before plotting
                # flatten if needed is handled by visualizer, but let's be safe on input type
                try:
                    visualizer.plot(
                        latents, 
                        labels=labels, 
                        save_path=save_path, 
                        title=f"{self.model_name} - {key} ({method.upper()})",
                        show=False
                    )
                except Exception as e:
                    print(f"Error plotting latent viz {key}: {e}")

        if fig:
            plt.close(fig)
        return fig

    def _plot_tensor(self, ax, tensor: torch.Tensor, label: str, config: Dict):
        data = tensor.numpy()
        x = np.arange(len(data))
        types = config["types"]
        params = config["params"]

        # Handle x scaling
        if "x_scale" in params:
            x = x * params["x_scale"]
        if "x_start" in params or "x_end" in params:
            mask = np.ones_like(x, dtype=bool)
            if "x_start" in params:
                mask &= x >= params["x_start"]
            if "x_end" in params:
                mask &= x <= params["x_end"]
            x, data = x[mask], data[mask]

        if PlotType.BAR not in types:
             ax.plot(x, data, label=label)

        # Rolling average
        if PlotType.ROLLING_AVG in types:
            window = params.get("rolling_window", 10)
            if window > 1 and len(data) >= window:
                roll = np.convolve(data, np.ones(window) / window, mode="valid")
                ax.plot(x[window - 1 :], roll, label=f"{label} (rolling {window})")

        if PlotType.EXPONENTIAL_AVG in types:
            # The 'beta' parameter controls the smoothness of the EMA.
            # A common range is 0.9 to 0.999.
            # Higher beta means more smoothing (slower decay).
            beta = params.get("ema_beta", 0.9)

            if len(data) > 0:
                ema_data = np.zeros_like(data)
                v = 0.0  # Initial EMA value

                # TensorBoard often applies bias correction, but for simplicity
                # and common practice, we'll implement the standard EMA (v = beta*v + (1-beta)*datum)
                # and initialize v with the first data point.
                v = data[0]
                ema_data[0] = v

                for i in range(1, len(data)):
                    v = beta * v + (1 - beta) * data[i]
                    ema_data[i] = v

                ax.plot(
                    x, ema_data, label=f"{label} (EMA $\\beta={beta}$)", linestyle="-."
                )

        # Variation fill (std dev)
        if PlotType.VARIATION_FILL in types:
            mean = np.mean(data)
            std = np.std(data)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, label=f"{label} ±σ")

        # Best fit line
        if PlotType.BEST_FIT_LINE in types and len(x) > 1:
            coeffs = np.polyfit(x, data, 1)
            fit = np.polyval(coeffs, x)
            ax.plot(x, fit, linestyle="--", label=f"{label} fit")

        # Bar chart for the last step
        if PlotType.BAR in types:
            # We want to plot the values of the *last* step as a bar chart.
            # 'data' here is likely a 2D array: (steps, num_classes) or 1D (steps,)
            # But the 'append' logic for vector stats creates a 2D tensor: (steps, vector_dim).
            # When we pull it out as numpy, it's (steps, vector_dim).
            
            # If we are doing a bar chart, we assume we want to visualize the distribution
            # at the latest time step.
            if len(data.shape) > 1:
                latest_data = data[-1]
                indices = np.arange(len(latest_data))
                ax.bar(indices, latest_data, label=f"{label} (latest)")
                ax.set_xticks(indices)
                ax.set_xlabel("Index")
            else:
                # If it's just scalar data over time, a bar chart might mean
                # plotting the history as bars? Usually not what's intended if combined with line plots.
                # But let's support plotting the history as bars if requested for 1D data.
                ax.bar(x, data, label=label)


        # Logarithmic scale
        if PlotType.LOG_Y in types:
            ax.set_yscale("log")

        ax.set_title(f"{self.model_name} - {label}")
        ax.set_xlabel("Steps")
        ax.set_ylabel(label)
        ax.grid()
        ax.legend()

    def get_data(self):
        if self._is_client:
            return None
        return self.stats
