import numpy as np
import torch
import torch.multiprocessing as mp
from queue import Empty
from typing import Dict, Optional, List, Any
import matplotlib.pyplot as plt
from enum import Enum, auto


class PlotType(Enum):
    ROLLING_AVG = auto()
    VARIATION_FILL = auto()
    BEST_FIT_LINE = auto()
    LOG_Y = auto()
    EXPONENTIAL_AVG = auto()


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
            if isinstance(self.stats[key], Dict):
                if subkey is None:
                    raise ValueError(f"Stat '{key}' requires a subkey")
                self.stats[key][subkey] = torch.cat(
                    (self.stats[key][subkey], torch.tensor([value]))
                )
            else:
                self.stats[key] = torch.cat((self.stats[key], torch.tensor([value])))

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

        plt.tight_layout()
        if dir:
            plt.savefig(f"{dir}/{self.model_name}_stats.png")
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
