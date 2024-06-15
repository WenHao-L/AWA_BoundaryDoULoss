import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


def sigmod_range(l=0., r=1.):

    def get_activation(left, right):

        def activation(x):

            return torch.sigmoid(x) * (right - left) + left

        return activation

    return get_activation(l, r)


class BaseConv(nn.Module):
    """A Conv3d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, bias=True):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size=ksize,
                              stride=stride,
                              padding=pad,
                              bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class AWModule(nn.Module):
    def __init__(self, in_ch=1, gamma_range=[0., 1.], use_checkpoint=False):
        super().__init__()
        nf = in_ch * 8
        self.gamma_range = gamma_range
        self.use_checkpoint = use_checkpoint

        self.head1 = BaseConv(in_ch, nf, ksize=3, stride=2)
        self.body1 = BaseConv(nf, nf*2, ksize=3, stride=2)
        self.body2 = BaseConv(nf*2, nf*4, ksize=3, stride=2)
        self.body3 = BaseConv(nf*4, nf*2, ksize=3)
        self.pooling = nn.AdaptiveAvgPool3d(1)

        self.image_adaptive_window_level = nn.Sequential(
            nn.Linear(nf*2, nf*4),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(nf*4, in_ch, bias=False)
        )

        self.image_adaptive_window_width = nn.Sequential(
            nn.Linear(nf*2, nf*4),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(nf*4, in_ch, bias=False)
        )

    def apply_window(self, img, params_level, params_width):
        params_level = sigmod_range(0, 1)(params_level)[..., None, None, None]
        params_width = sigmod_range(0, 1)(params_width)[..., None, None, None]
        img = (img - params_level + params_width / 2) / (params_width + 0.0001)
        img = torch.clamp(img, 0, 1)        
        return img

    def forward_part(self, img):

        fea = self.head1(img)
        fea_s2 = self.body1(fea)
        fea_s4 = self.body2(fea_s2)
        fea_s8 = self.body3(fea_s4)

        fea_window = self.pooling(fea_s8)
        fea_window = fea_window.view(fea_window.shape[0], fea_window.shape[1])

        para_window_level = self.image_adaptive_window_level(fea_window)
        para_window_width = self.image_adaptive_window_width(fea_window)
        img_window = self.apply_window(img, para_window_level, para_window_width)

        return img_window
    
    def forward(self, img):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self.forward_part, img)
        else:
            return self.forward_part(img)

if __name__ == "__main__":
    from thop import profile

    input = torch.randn(1, 1, 96, 96, 96)
    model = AWModule(in_ch=1).to(device='cpu')

    flops, params = profile(model, inputs=(input,))
    print('FLOPs = {:.2f}G'.format(flops/1000**3))
    print('Params = {:.2f}M'.format(params/1000**2))
    print('FLOPs = {:.2f}'.format(flops))
    print('Params = {:.2f}'.format(params))
