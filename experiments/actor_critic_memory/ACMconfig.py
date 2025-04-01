import torch

class ACModelconfig():
    """
    Configuration class for the ACM model (general).
    Contains configurations for PPO, ACM network, memory buffer, and ACM agent.
    """

    def __init__(self, config=None):
        """
        Initialize the configuration class.
        Args:
            env (str): Environment name
            PPOconfig (Config): Configuration object for PPO
            ACMNconfig (Config): Configuration object for ACM network
            Buffconfig (Config): Configuration object for memory buffer
            ACMconfig (Config): Configuration object for ACM agent
        """
        self.PPOconfig = config["PPOconfig"]
        self.ACMNconfig = config["ACMNconfig"]
        self.Buffconfig = config["Buff"]
        self.ACMconfig = config["ACMconfig"]
        self.MHABconfig = config["MHABconfig"]


class ACMNconfig():
    """
    Configuration class for the ACM network.
    Network = Attention based network (Contains Q, K, V matrices)
    """

    def __init__(self, config=None):
        """
        Initialize the configuration class.
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            embed_shapes (torch.Tensor): Embedding shapes
            log (bool): Log flag
            optimizer (str): Optimizer type
            learning_rate (float): Learning rate
        """
        self.input_dim = config["input_shape"]
        self.output_dim = config["output_shape"]
        self.log = config["log"]
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.embed_shapes = config["embed_shapes"]
        self.device = config["device"]
        self.adam_epsilon = config["adam_epsilon"]
        self.clipnorm = config["clipnorm"]
        

class ACMAconfig():
    """
    Configuration class for the ACM agent.
    Agent = Generalized Actor-Critic Memory (Contains PPO, ACM Network, Memory Buffer)
    """

    def __init__(self, config=None):
        """
        Initialize the configuration class.
        Args:
            
        """
        self.env = config["env"]
        self.checkpoint_path = config["checkpoint_path"]
        self.training_steps = config["training_steps"]
        self.print_interval = config["print_interval"]
        self.steps_per_epoch = config["steps_per_epoch"]
        self.checkpoint_interval = config["checkpoint_interval"]
        self.logging = config["log"]
        

class Buffconfig():
    """
    Configuration class for the memory buffer.
    """

    def __init__(self, config=None):
        """
        Initialize the configuration class.
        Args:
            buffer_size (int): Buffer size
            batch_size (int): Batch size
            history_length (int): History length
        """
        self.buffer_size = config["buffer_size"]


class MHABconfig():
    """
    Configuration class for the multi-head attention block.
    """

    def __init__(self, config=None):
        """
        Initialize the configuration class.
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            num_heads (int): Number of heads
            dropout (float): Dropout rate
        """
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.adam_epsilon = config["adam_epsilon"]
        self.clipnorm = config["clipnorm"]
        self.layers = config["layers"]
        self.self_attention = config["self_attention"]

        # CHECKING LAYERS
        for layer in self.layers:
            assert layer["type"] in ["MultiheadAttention", "Linear", "ReLU", "Dropout", "LayerNorm", "Softmax"], "Invalid layer type"
            if layer["type"] == "MultiheadAttention":
                assert "num_heads" in layer, "num_heads is required for MultiheadAttention"
                assert "dropout" in layer, "dropout is required for MultiheadAttention"
                assert "embed_dim" in layer, "embed_dim is required for MultiheadAttention"
                assert "bias" in layer, "bias is required for MultiheadAttention"
                assert "batch_first" in layer, "batch_first is required for MultiheadAttention"
                assert "kdim" in layer, "kdim is required for MultiheadAttention"
                assert "vdim" in layer, "vdim is required for MultiheadAttention"
                assert "add_zero_attn" in layer, "add_zero_attn is required for MultiheadAttention"
                assert "add_bias_kv" in layer, "add_bias_kv is required for MultiheadAttention"
            elif layer["type"] == "Linear":
                assert "in_features" in layer, "in_features is required for Linear"
                assert "out_features" in layer, "out_features is required for Linear"
                assert "bias" in layer, "bias is required for Linear"
            elif layer["type"] == "Softmax":
                assert "dim" in layer, "dim is required for Softmax"
            elif layer["type"] == "Dropout":
                assert "p" in layer, "p is required for Dropout"
            elif layer["type"] == "LayerNorm":
                assert "normalized_shape" in layer, "normalized_shape is required for LayerNorm"
                assert "eps" in layer, "eps is required for LayerNorm"
                assert "elementwise_affine" in layer, "elementwise_affine is required for LayerNorm"
        