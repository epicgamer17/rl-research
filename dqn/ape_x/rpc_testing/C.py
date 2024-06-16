import torch.distributed.rpc as rpc

class Class:
    def __init__(self, stop_fn):
        self.stop_fn=stop_fn
        pass
    
    def do_stop(self, info):
        return rpc.remote(info, self.stop_fn, (True,))
