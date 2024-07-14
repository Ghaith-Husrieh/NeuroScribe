class Dispatcher:
    _backends = {}
    _default_backends = {'cpu', 'cuda'}
    _custom_backends = set()

    @classmethod
    def register_backend(cls, device, backend_class):
        device = cls._validate_and_normalize_device(device)
        if device in cls._default_backends and device in cls._backends:
            raise ValueError(f"Cannot override the default backend implementation for device: '{device}'.")
        cls._backends[device] = backend_class
        if device not in cls._default_backends:
            cls._custom_backends.add(device)

    @classmethod
    def get_backend(cls, device):
        device = cls._validate_and_normalize_device(device)
        if device not in cls._backends:
            if device in cls._default_backends:
                raise ValueError(f"NeuroScribe is not installed with support for {device.upper()} devices.")
            else:
                raise ValueError(
                    f"Unsupported device '{device}'. Supported devices are: {
                        cls._default_backends.union(cls._custom_backends)}."
                )
        return cls._backends[device]

    @staticmethod
    def _validate_and_normalize_device(device):
        if not isinstance(device, str):
            raise TypeError(f"The 'device' argument must be a string. Received '{type(device)}' instead.")
        return device.lower()


from neuroscribe.core._tensor_lib.backend.cpu.cpu_backend import CPUBackend

Dispatcher.register_backend('cpu', CPUBackend)

try:
    from neuroscribe.core._tensor_lib.backend.cuda.cuda_backend import \
        CUDABackend
    Dispatcher.register_backend('cuda', CUDABackend)
except ImportError:
    pass
