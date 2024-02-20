import torch.nn as nn
from architectures.linear_sequential import linear_sequential
from architectures.convolution_sequential import convolution_sequential

class ConvLinSeq(nn.Module):
    def __init__(self, input_dims, linear_hidden_dims, conv_hidden_dims, output_dim, kernel_dim, k_lipschitz, p_drop):
        super().__init__()
        if k_lipschitz is not None:
            k_lipschitz = k_lipschitz ** (1./2.)
        self.convolutions = convolution_sequential(input_dims=input_dims,
                                                   hidden_dims=conv_hidden_dims,
                                                   output_dim=output_dim,
                                                   kernel_dim=kernel_dim,
                                                   k_lipschitz=k_lipschitz,
                                                   p_drop=p_drop)
        # We assume that conv_hidden_dims is a list of same hidden_dim values
        self.linear = linear_sequential(input_dims=[conv_hidden_dims[-1] * (input_dims[0] // 2 ** len(conv_hidden_dims)) * (input_dims[1] // 2 ** len(conv_hidden_dims))],
                                        hidden_dims=linear_hidden_dims,
                                        output_dim=output_dim,
                                        k_lipschitz=k_lipschitz,
                                        p_drop=p_drop)

    def forward(self, input):
        batch_size = input.size(0)
        input = self.convolutions(input)
        input = self.linear(input.view(batch_size, -1))
        return input


def convolution_linear_sequential(input_dims, linear_hidden_dims, conv_hidden_dims, output_dim, kernel_dim, k_lipschitz=None, p_drop=None):
    return ConvLinSeq(input_dims=input_dims,
                      linear_hidden_dims=linear_hidden_dims,
                      conv_hidden_dims=conv_hidden_dims,
                      output_dim=output_dim,
                      kernel_dim=kernel_dim,
                      k_lipschitz=k_lipschitz,
                      p_drop=p_drop)
