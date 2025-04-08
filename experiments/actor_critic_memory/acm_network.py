# Base attention network, no self attention

import torch
import matplotlib.pyplot as plt

class ACMNetwork():
    def __init__(self, config=None, device="cpu"):
        """"
        # Base attention network (IMPLEMENTED BY HAND FOR TESTING, IMPLEMENTED IN PYTORCH UNDER : MULTIHEAD ATTENTION)
        Args:
            config (Config): Configuration object
            device (str): Device to use ("cpu" or "cuda")
        """
        # Parse config
        assert config is not None, "Config must be provided"
        assert device is not None, "Device must be provided"
        assert len(config.embed_shapes) == 3, "NEED EMBED SHAPES FOR Q, K, V (disable for precomputed QKV models)"
        self.input_dim = config.input_dim
        self.output_dim = config.embed_shapes[2]
        embed_shapes = config.embed_shapes
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.epsilon = config.adam_epsilon
        self.clipnorm = config.clipnorm
        self.Qdim = embed_shapes[0]
        self.Kdim = embed_shapes[1]
        self.Vdim = embed_shapes[2]
        self.logging = config.log
        self.device = device

        if self.logging:
            print("ACMNetwork: input_dim: {}, output_dim: {}, Qdim: {}, Kdim: {}, Vdim: {}".format(self.input_dim, self.output_dim, self.Qdim, self.Kdim, self.Vdim))
            print("ACMNetwork: optimizer: {}, learning_rate: {}".format(self.optimizer, self.learning_rate))

        # LAYERS INPUT : [(type Layer config),  )   ]



        # Initialize Key, Value and Query matrices
        WQ = torch.zeros((self.input_dim, self.Qdim), dtype=torch.float32).to(self.device)
        WK = torch.zeros((self.input_dim, self.Kdim), dtype=torch.float32).to(self.device)
        WV = torch.zeros((self.input_dim, self.Vdim), dtype=torch.float32).to(self.device)

        if self.logging:
            print("ACMNetwork: WQ shape: {}, WK shape: {}, WV shape: {}".format(WQ.shape, WK.shape, WV.shape))
        

        WQ = torch.nn.Parameter(WQ, requires_grad=True)
        WK = torch.nn.Parameter(WK, requires_grad=True)
        WV = torch.nn.Parameter(WV, requires_grad=True)
        self.WQ = WQ
        self.WK = WK
        self.WV = WV
        torch.nn.init.xavier_uniform_(self.WQ)
        torch.nn.init.xavier_uniform_(self.WK)
        torch.nn.init.eye_(self.WV)

        # Testing shapes
        input = torch.randn(self.input_dim).to(self.device)
        assert torch.matmul(input, self.WQ).shape == (self.Qdim,), "Q shape mismatch! input_dim: {}, Qdim: {}, outdim: {}".format(input.shape, self.WQ.shape, torch.matmul(self.WQ, input).shape)
        assert torch.matmul(input, self.WK).shape == (self.Kdim,), "K shape mismatch! input_dim: {}, Kdim: {}, outdim: {}".format(input.shape, self.WK.shape, torch.matmul(self.WK, input).shape)
        assert torch.matmul(input, self.WV).shape == (self.Vdim,), "V shape mismatch! input_dim: {}, Vdim: {}, outdim: {}".format(input.shape, self.WV.shape, torch.matmul(self.WV, input).shape)

        # Initialize optimizer
        if self.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(params=[self.WQ, self.WK, self.WV], lr=self.learning_rate)
        elif self.optimizer == "Adam":
            #, self.WV
            self.optimizer = torch.optim.Adam(params=[self.WQ, self.WK, self.WV], lr=self.learning_rate, eps=self.epsilon)
        elif self.optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(params=[self.WQ, self.WK, self.WV], lr=self.learning_rate)
        elif self.optimizer == "Adagrad":
            self.optimizer = torch.optim.Adagrad(params=[self.WQ, self.WK, self.WV], lr=self.learning_rate)
            

    def forward(self, state, history, flag=False):
        """
        Forward pass of the network
        Args:
            state (torch.Tensor): Current state, shape (input_dim,)
            history (torch.Tensor): History of states, shape (history_length, input_dim)
        Returns:
            torch.Tensor: Output of the network
        """
        assert state is not None, "State must be provided"
        assert history is not None, "History must be provided"
        assert len(state.shape) == 2, "State must be of shape (1xinput_dim), got {}".format(state.shape)
        assert len(history.shape) == 2, "History must be of shape (history_length x input_dim), got {}".format(history.shape)
        assert state.shape[1] == self.input_dim, "State must be of shape (1xinput_dim), got {}".format(state.shape)
        assert history.shape[1] == self.input_dim, "History must be of shape (history_length x input_dim), got {}".format(history.shape)


        # Compute Q, K, V
        # print("State: {}".format(state))
        # print("History: {}".format(history))
        Q = torch.matmul(state, self.WQ)
        K = torch.matmul(history, self.WK)
        V = torch.matmul(history, self.WV)

        # print("Q: {}".format(Q))
        # print("K: {}".format(K))
        # print("V: {}".format(V))

        # Check shapes
        # Q should be of size 1 x Qdim
        # K and V should be of size history_length x Kdim and history_length x Vdim
        assert Q.shape == (1, self.Qdim), "Q shape mismatch! Q: {}, Qdim: {}".format(Q.shape, self.Qdim)
        assert K.shape == (history.shape[0], self.Kdim), "K shape mismatch! K: {}, Kdim: {}".format(K.shape, self.Kdim)
        assert V.shape == (history.shape[0], self.Vdim), "V shape mismatch! V: {}, Vdim: {}".format(V.shape, self.Vdim)

        # Compute attention weights
        temp = 0.1
        attn_scores = torch.matmul(Q, K.T) / (self.Kdim ** 0.5) / temp

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Plot attention weights as grid, plot Q, K, V as well
        if flag==True:
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(Q.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("Q")
            plt.xlabel("State")
            plt.ylabel("Qdim")
            plt.subplot(2, 2, 2)
            plt.imshow(K.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("K")
            plt.xlabel("History")
            plt.ylabel("Kdim")
            plt.subplot(2, 2, 3)
            plt.imshow(V.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("V")
            plt.xlabel("History")
            plt.ylabel("Vdim")
            plt.subplot(2, 2, 4)
            plt.imshow(attn_weights.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("Attention Weights")
            plt.xlabel("History")
            plt.ylabel("State")
            plt.show()
            print("Attention Weights: {}".format(attn_weights))
            print("Values: {}".format(V))
            print("OUTPUT: {}".format(torch.matmul(attn_weights, V)))
            


        

        # Compute output
        output = torch.matmul(attn_weights, V)
        # print("Output: {}".format(output))
        # Check output shape
        # Output should be of size 1 x Vdim

        print("Input: {}".format(state))
        print("History: {}".format(history))
        print("Q: {}".format(Q))
        print("K: {}".format(K))
        print("V: {}".format(V))
        print("Attention Weights: {}".format(attn_weights))
        print("Attention Scores: {}".format(attn_scores))
        print("Output: {}".format(output))
        assert output.shape == (1, self.Vdim), "Output shape mismatch! Output: {}, Vdim: {}".format(output.shape, self.Vdim)

        return output
    
    def get_weights(self):
        """
        Get weights of the network
        Returns:
            int: eigenvalues of WQ weights
            int: eigenvalues of WK weights
            int: eigenvalues of WV weights
        """
        # if square matrices then you can do eigvals
        if self.WQ.shape[0] == self.WQ.shape[1] and self.WK.shape[0] == self.WK.shape[1] and self.WV.shape[0] == self.WV.shape[1]:
            WQ_norm = torch.linalg.eigvals(self.WQ)
            WK_norm = torch.linalg.eigvals(self.WK)
            WV_norm = torch.linalg.eigvals(self.WV)
        else:
            # else get mean of weights
            WQ_norm = torch.mean(self.WQ)
            WK_norm = torch.mean(self.WK)
            WV_norm = torch.mean(self.WV)
        
        # print("WQ GRAD: {}".format(self.WQ.grad))
        # print("WK GRAD: {}".format(self.WK.grad))

        # print("WQ: {}".format(self.WQ))
        # print("WK: {}".format(self.WK))


        return WQ_norm, WK_norm, WV_norm
    
    def learn(self, loss):
        """
        Learn from the loss
        Args:
            loss (torch.Tensor): Loss to learn from
        """
        # print("Learning from loss: {}".format(loss))
        # print("BEFORE LEARNING")
        # print("WQ: {}".format(self.WQ))
        # print("WK: {}".format(self.WK))
        # print("WV: {}".format(self.WV))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("AFTER LEARNING")
        # print("WQ: {}".format(self.WQ))
        # print("WK: {}".format(self.WK))
        # print("WV: {}".format(self.WV))
        if self.logging:
            print("Loss: {}".format(loss.item()))
            print("WQ: {}".format(self.WQ))
            print("WK: {}".format(self.WK))
            print("WV: {}".format(self.WV))
        if self.logging:
            WQ_eigenvalues, WK_eigenvalues, WV_eigenvalues = self.get_weights()
            print("WQ eigenvalues: {}".format(WQ_eigenvalues))
            print("WK eigenvalues: {}".format(WK_eigenvalues))
            print("WV eigenvalues: {}".format(WV_eigenvalues))


        


class MHA(torch.nn.Module):
    def __init__(self, config=None, device="cpu", logging=False, child=False):
        """
        Multiheaded attention network block
        Args:
            input_dim (int): Input dimension
            embed_shapes (list): List of shapes for Q, K, V
            device (str): Device to use ("cpu" or "cuda" or mps)
        """
        super().__init__()

        # Parse config
        assert config is not None, "Config must be provided"
        assert device is not None, "Device must be provided"
        
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.epsilon = config.adam_epsilon
        self.clipnorm = config.clipnorm
        self.logging = logging
        self.device = device
        self.self_attention = config.self_attention
        self.layers = torch.nn.ModuleList([
            (torch.nn.MultiheadAttention(embed_dim=layer["embed_dim"], 
                                        num_heads=layer["num_heads"], 
                                        dropout=layer["dropout"], 
                                        bias=layer["bias"], 
                                        add_bias_kv=layer["add_bias_kv"], 
                                        add_zero_attn=layer["add_zero_attn"], 
                                        kdim=layer["kdim"],
                                        vdim=layer["vdim"], 
                                        batch_first=layer["batch_first"], 
                                        device=self.device, 
                                        dtype=torch.float32) 
            if layer["type"] == "MultiheadAttention" else
            torch.nn.Linear(in_features=layer["in_features"], 
                            out_features=layer["out_features"], 
                            bias=layer["bias"])
            if layer["type"] == "Linear" else
            torch.nn.ReLU()
            if layer["type"] == "ReLU" else
            torch.nn.Softmax(dim=layer["dim"])
            if layer["type"] == "Softmax" else
            torch.nn.Dropout(p=layer["p"])
            if layer["type"] == "Dropout" else
            torch.nn.LayerNorm(normalized_shape=layer["normalized_shape"], 
                                eps=layer["eps"], 
                                elementwise_affine=layer["elementwise_affine"], 
                                device=self.device, 
                                dtype=torch.float32)
            if layer["type"] == "LayerNorm" else
            print("Unknown layer type: {}".format(layer["type"]))
        ) for layer in config.layers
        ])

        # Initialize optimizer
        if not child:
            if self.optimizer == "SGD":
                self.optimizer = torch.optim.SGD(params=self.parameters(), lr=self.learning_rate)
            elif self.optimizer == "Adam":
                self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, eps=self.epsilon)
            elif self.optimizer == "RMSprop":
                self.optimizer = torch.optim.RMSprop(params=self.parameters(), lr=self.learning_rate)
            elif self.optimizer == "Adagrad":
                self.optimizer = torch.optim.Adagrad(params=self.parameters(), lr=self.learning_rate)

        if self.logging:
            print("MHA: optimizer: {}, learning_rate: {}".format(self.optimizer, self.learning_rate))
            print("MHA: layers: {}".format(self.layers))

        # Check if device is available
        if not child:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            # elif torch.backends.mps.is_available():
            #     self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        if self.logging:
            print("MHA: device: {}".format(self.device))

        # # Move model to device
        # self.to(self.device)

        if self.logging:
            print("MHA: model is on device: {}".format(self.device))

    def forward(self, state, history):
        """
        Forward pass of the network
        Args:
            state (torch.Tensor): Current state, shape (input_dim,)
            history (torch.Tensor): History of states, shape (history_length, input_dim)
        Returns:
            torch.Tensor: Concatenated output of the State + Augmented past
        """

        assert state is not None, "State must be provided"
        assert history is not None, "History must be provided"
        # SELF ATTENTION VS CROSS CHECK THE DIMENSIONS OR SOMETHING TO DO
        # TO DO : ADD RESIDUAL CONNECTION POSSIBILITY
        # ADD BETTER LOGGING AND BETTER SHAPE CHECKING AND MANIPULATION
        # IF BATCH_FIRST == TRUE THEN IT IS OF SHAPE : BATCH X SEQ X EMBED
        # IF BATCH_FIRST == FALSE THEN IT IS OF SHAPE : SEQ X BATCH X EMBED
        x = history
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, torch.nn.MultiheadAttention):
                # self attention
                if self.self_attention[i]==True:
                    # Self attention
                    if self.logging:
                        print(x)
                        print("CURRENT SHAPE: {}".format(x.shape))
                        print("LAYER: {}".format(layer))
                    attn_out, attnweights = layer(query=x.unsqueeze(0), key=x.unsqueeze(0), value=x.unsqueeze(0))
                    if self.logging:
                        print("X shape: {}".format(x.unsqueeze(0)))
                        print("ATTN WEIGHTS: {}".format(attn_out))
                        print("ATTN WEIGHTS: {}".format(attn_out[0][0]))
                    x = attn_out[0]
                else:
                    if self.logging:
                        print("ATTENTION")
                        print("QDIM: {}".format(layer.embed_dim))
                        print("KDIM: {}".format(layer.kdim))
                        print("VDIM: {}".format(layer.vdim))
                        print("CURRENT SHAPE: {}".format(x.shape))
                        print("STATE SHAPE: {}".format(state.shape))
                        print("LAYER: {}".format(layer))
                        print("BATCH FIRST: {}".format(layer.batch_first))
                        print("SQUEEZED STATE: {}".format(state.unsqueeze(0)))
                        print("SQUEEZE SHAPE: {}".format(state.unsqueeze(0).shape))
                        print("SQUEEZED X: {}".format(x.unsqueeze(0)))
                        print("SQUEEZE SHAPE: {}".format(x.unsqueeze(0).shape))
                    attn_out, attnweights = layer(query=state.unsqueeze(0), key=x.unsqueeze(0), value=x.unsqueeze(0))
                    if self.logging:
                        print("X shape: {}".format(x.unsqueeze(0)))
                        print("ATTN WEIGHTS: {}".format(attn_out))
                        print("ATTN WEIGHTS: {}".format(attn_out[0][0]))
                    x = attn_out[0]
            else:
                if self.logging:
                    print(x)
                    print("CURRENT SHAPE: {}".format(x.shape))
                    print("LAYER: {}".format(layer))
                x = layer(x)
                if self.logging:
                    print("AFTER LAYER: {}".format(x))
                    print("AFTER LAYER SHAPE: {}".format(x.shape))
        if self.logging:
            # Concatenate state and history
            print("FINAL X: {}".format(x))
            print("FINAL STATE: {}".format(state))
            # Check shape
            print("x shape: {}".format(x.shape))
            print("state shape: {}".format(state.shape))
            # check effect of view
            print("state view shape: {}".format(state.view(-1).shape))

        x = torch.cat((state, x), dim=-1)
        # Check shape
        #should be dim 1 x (value_dim of last mhalayer + input_dim)
        if self.logging:
            print("OUTPUT SHAPE: {}".format(x.shape))

        return x
    
    def learn(self, loss):
        """
        Learn from the loss
        Args:
            loss (torch.Tensor): Loss to learn from
        """
        # print("Learning from loss: {}".format(loss))
        # print("BEFORE LEARNING")
        # print("WQ: {}".format(self.WQ))
        # print("WK: {}".format(self.WK))
        # print("WV: {}".format(self.WV))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()