import torch
import torch.nn as nn
import numpy as np

class ScaledSigmoid(nn.Module):
    def __init__(self, upbound_window=255.):
        super(ScaledSigmoid, self).__init__()
        self.upbound_window = upbound_window

    def forward(self, x):
        return self.upbound_window * nn.Sigmoid()(x)


class ScaledReLU(nn.Module):
    def __init__(self, upbound_window=255.):
        super(ScaledReLU, self).__init__()
        self.upbound_window = upbound_window

    def forward(self, x):
        return torch.clamp(nn.Sigmoid()(x), max=self.upbound_window)


class WindowOptimizer(nn.Module):
    def __init__(self, out_channels, act_window, upbound_window=255., window_init=None, mode="2D"):
        super(WindowOptimizer, self).__init__()
        if mode == "2D":
            self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            if window_init is not None:
                w_conv = np.zeros((out_channels, 1, 1, 1), dtype=np.float32)
                b_conv = np.zeros(out_channels, dtype=np.float32)
                for idx in range(out_channels):
                    wl, ww = window_init[idx]
                    w_new, b_new = self.get_init_conv_params(wl, ww, act_window, upbound_window)
                    w_conv[idx,0,0,0] = w_new
                    b_conv[idx] = b_new
                w_conv = torch.from_numpy(w_conv)
                b_conv = torch.from_numpy(b_conv)
                self.conv.weight = nn.Parameter(w_conv)
                self.conv.bias = nn.Parameter(b_conv)
        elif mode == "3D":
            self.conv = nn.Conv3d(in_channels=1, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            if window_init is not None:
                w_conv = np.zeros((out_channels, 1, 1, 1, 1), dtype=np.float32)
                b_conv = np.zeros(out_channels, dtype=np.float32)
                for idx in range(out_channels):
                    wl, ww = window_init[idx]
                    w_new, b_new = self.get_init_conv_params(wl, ww, act_window, upbound_window)
                    w_conv[idx,0,0,0,0] = w_new
                    b_conv[idx] = b_new
                w_conv = torch.from_numpy(w_conv)
                b_conv = torch.from_numpy(b_conv)
                self.conv.weight = nn.Parameter(w_conv)
                self.conv.bias = nn.Parameter(b_conv)
        else:
            raise Exception()

        if act_window == "relu":
            self.act = ScaledReLU(upbound_window)
        elif act_window == "sigmoid":
            self.act = ScaledSigmoid(upbound_window)
        else:
            raise Exception()

    def get_init_conv_params(self, wl, ww, act_window, upbound_value=255.):
        if act_window == 'sigmoid':
            w_new, b_new = self.get_init_conv_params_sigmoid(wl, ww, upbound_value=upbound_value)
        elif act_window == 'relu':
            w_new, b_new = self.get_init_conv_params_relu(wl, ww, upbound_value=upbound_value)
        else:
            raise Exception()
        return w_new, b_new

    def get_init_conv_params_relu(self, wl, ww, upbound_value=255.):
        w = upbound_value / ww
        b = -1. * upbound_value * (wl - ww / 2.) / ww
        return (w, b)

    def get_init_conv_params_sigmoid(self, wl, ww, upbound_value=255., smooth=None):
        if smooth is None:
            smooth = upbound_value / 255.0

        w = 2./ww * np.log(upbound_value/smooth - 1.)
        b = -2.*wl/ww * np.log(upbound_value/smooth - 1.)
        return (w, b)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


if __name__ == "__main__":
    out_channels = 1
    act_window = 'relu'
    upbound_window = 1
    window_init = [(0.5, 0.5) for _ in range(out_channels)]

    window_optimizer = WindowOptimizer(out_channels=out_channels, act_window=act_window, upbound_window=upbound_window, window_init=window_init, mode="3D")

    total_params = sum(p.numel() for p in window_optimizer.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")