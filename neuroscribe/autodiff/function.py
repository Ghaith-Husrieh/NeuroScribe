class Function:
    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self, result_tensor): raise RuntimeError(f"backward not implemented for {type(self)}")
