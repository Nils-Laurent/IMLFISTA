import numpy
import torch.nn
import deepinv
"""
Created on Oct 18 2024
@author: Nils Laurent
"""

class DownsamplingTransfer:
    def __init__(self, def_filter, padding="circular", factor=2):
        self.filter_object = def_filter
        self.padding = padding
        self.factor = factor
        self.op = None  # will be created on first use

    def _initialize_operator(self, x, target_shape):
        device = x.device
        dtype = x.dtype

        if isinstance(self.filter_object, SincFilter):
            filt_2d = self.filter_object.get_filter_2d().to(device)
        else:
            k0 = self.filter_object.get_filter()
            filt_2d = self.set_2d_filter(k0, dtype).to(device)

        padding = "valid" if isinstance(self.filter_object, Dirac) else self.padding

        self.op = deepinv.physics.Downsampling(
            target_shape, filter=filt_2d, factor=self.factor, device=device, padding=padding
        )
    def _getfilter(self):
        return self.set_2d_filter(self.filter_object.get_filter(), torch.float32)
    
    def set_2d_filter(self, k0, dtype):
        k_filter = torch.outer(k0, k0).unsqueeze(0).unsqueeze(0).type(dtype)
        return k_filter

    def to_coarse(self, x, target_shape):
        if self.op is None:
            self._initialize_operator(x, target_shape)

        if x.dim() == 3:
            return self.op.A(x.unsqueeze(0)).squeeze(0)
        return self.op.A(x)

    def to_fine(self, x, target_shape):
        if self.op is None:
            self._initialize_operator(x, target_shape)

        if isinstance(self.filter_object, Dirac):
            upsample = torch.nn.Upsample(scale_factor=self.factor, mode='nearest')
            return upsample(x)
        return self.op.A_adjoint(x) * self.factor ** 2
    def to(self, device):
        # Just in case someone tries to move it like a model
        if self.op is not None:
            self.op.to(device)
        return self


# ==========================
#       filter list
# ==========================


class Kaiser:
    def __str__(self):
        return 'kaiser'

    def get_filter(self):
        # N = 10
        # beta = 10.0
        k0 = torch.tensor([
            0.0004, 0.0310, 0.2039, 0.5818, 0.9430,
            0.9430, 0.5818, 0.2039, 0.0310, 0.0004
        ])
        return k0

class SincFilter:
    def __str__(self):
        return 'sinc'

    def get_filter_2d(self):
        return deepinv.physics.blur.sinc_filter(factor=2, length=11, windowed=True)

    def get_filter(self):
        f = self.get_filter_2d()
        return f[0, 0, 5, :]

class CFir:  # custom FIR filter
    def __str__(self):
        return 'cfir'

    def get_filter(self):
        # order + 1 coefficients
        k0 = torch.tensor([
            -0.015938026, 0.000019591, 0.013033937, -0.000004666, -0.018657837, 0.000020187, 0.026570831, 0.000002218,
            -0.038348155, 0.000018390, 0.058441238, 0.000007421, -0.102893218, 0.000011707, 0.317258819, 0.500004593,
            0.317258819, 0.000011707, -0.102893218, 0.000007421, 0.058441238, 0.000018390, -0.038348155, 0.000002218,
            0.026570831, 0.000020187, -0.018657837, -0.000004666, 0.013033937, 0.000019591, -0.015938026
        ])
        #k0 = torch.tensor([
        #    -0.000068106, 0.111025515, 0.000061827, -0.103275087, 0.000049373, 0.317230919, 0.499913812, 0.317230919,
        #    0.000049373, -0.103275087, 0.000061827, 0.111025515, -0.000068106
        #])
        return k0


class BlackmannHarris:
    def __str__(self):
        return 'blackmannharris'

    def get_filter(self):
        # 8 coefficients
        #k0 = torch.tensor(
        #    [0.0001, 0.0334, 0.3328, 0.8894,
        #     0.8894, 0.3328, 0.0334, 0.0001]
        #)
        k0 = torch.tensor([
            3.9818e-05, 1.3299e-02, 1.3252e-01, 3.5415e-01, 3.5415e-01, 1.3252e-01, 1.3299e-02, 3.9818e-05]
        )
        return k0

class Dirac:
    def __str__(self):
        return 'dirac'

    def get_filter(self):
        k0 = torch.tensor([1.0])
        return k0
    
class Daubechies8:
    def __str__(self):
        return 'daubechies8'

    def get_filter(self):
        k0 = torch.tensor([0.2304,0.7148,0.6309,-0.0280,-0.1870,0.0308,0.0329,-0.0106])
        return k0

class Gaussian:
    def __str__(self):
        return 'gaussian'

    def get_filter(self):
        k0 = torch.tensor([0.0001, 0.0334, 0.3328, 0.8894,
                           0.8894, 0.3328, 0.0334, 0.0001])
        return k0

class Haar:
    def __str__(self):
        return 'haar'

    def get_filter(self):
        k0 = torch.tensor([1.0, 1.0])
        return k0
    
filter_classes = {
    "dirac": Dirac,
    "blackmannharris": BlackmannHarris,
    "cfir": CFir,
    "kaiser": Kaiser,
    "sinc": SincFilter,
    "gaussian": Gaussian,
    "daubechies8": Daubechies8,
    "haar": Haar,
}
def create_filter(name):
    if name in filter_classes:
        return filter_classes[name]()
    else:
        raise ValueError(f"Unknown filter type: {name}")
device = 'cpu'
filter = create_filter("blackmannharris")  # choose your filter
information_transfer = DownsamplingTransfer(filter)  # create operator
information_transfer = information_transfer.to(device)