import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from architectures.linear_sequential import linear_sequential
from architectures.convolution_linear_sequential import convolution_linear_sequential
from architectures.vgg_sequential import vgg16_bn


class ModifiedEvidentialNet(nn.Module):
    def __init__(self,
                 input_dims,  # Input dimension. list of ints
                 output_dim,  # Output dimension. int
                 hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
                 kernel_dim=None,  # Kernel dimension if conv architecture. int
                 architecture='linear',  # Encoder architecture name. int
                 k_lipschitz=None,  # Lipschitz constant. float or None (if no lipschitz)
                 batch_size=64,  # Batch size. int
                 lr=1e-3,  # Learning rate. float
                 loss='IEDL',  # Loss name. string
                 clf_type='softplus',
                 fisher_c=1.0,
                 lamb1=1.0,
                 lamb2=1.0,
                 seed=123):  # Random seed for init. int
        super().__init__()

        torch.cuda.manual_seed(seed)
        torch.set_default_tensor_type(torch.FloatTensor)

        # Architecture parameters
        self.input_dims, self.output_dim, self.hidden_dims, self.kernel_dim = input_dims, output_dim, hidden_dims, kernel_dim
        self.k_lipschitz = k_lipschitz
        # Training parameters
        self.batch_size, self.lr = batch_size, lr
        self.loss = loss

        # self.target_con = target_con
        # self.kl_c = kl_c
        self.target_con = 1.0
        self.kl_c = -1.0
        self.fisher_c = fisher_c
        self.lamb1 = lamb1
        self.lamb2 = lamb2

        self.loss_mse = torch.tensor(0.0)
        self.loss_var = torch.tensor(0.0)
        self.loss_kl = torch.tensor(0.0)
        self.loss_fisher = torch.tensor(0.0)

        # Feature selection
        if architecture == 'linear':
            self.sequential = linear_sequential(input_dims=self.input_dims,
                                                hidden_dims=self.hidden_dims,
                                                output_dim=self.output_dim,
                                                k_lipschitz=self.k_lipschitz)
        elif architecture == 'conv':
            assert len(input_dims) == 3
            self.sequential = convolution_linear_sequential(input_dims=self.input_dims,
                                                            linear_hidden_dims=self.hidden_dims,
                                                            conv_hidden_dims=[64, 64, 64],
                                                            output_dim=self.output_dim,
                                                            kernel_dim=self.kernel_dim,
                                                            k_lipschitz=self.k_lipschitz)
        elif architecture == 'vgg':
            assert len(input_dims) == 3
            self.sequential = vgg16_bn(output_dim=self.output_dim, k_lipschitz=self.k_lipschitz)
        else:
            raise NotImplementedError

        self.softmax = nn.Softmax(dim=-1)
        self.clf_type = clf_type

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if architecture == 'conv':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)

    def forward(self, input, labels_=None, return_output='alpha', compute_loss=False, epoch=10.):
        assert not (labels_ is None and compute_loss)

        # Forward
        logits = self.sequential(input)
        evidence = F.softplus(logits)
        alpha = evidence + self.lamb2

        # Calculate loss
        if compute_loss:
            labels_1hot = torch.zeros_like(logits).scatter_(-1, labels_.unsqueeze(-1), 1)
            self.loss_mse = self.compute_mse(labels_1hot, evidence)
            kl_alpha = evidence * (1 - labels_1hot) + self.lamb2
            self.loss_kl = self.compute_kl_loss(kl_alpha, self.lamb2)

            if self.kl_c == -1:
                regr = np.minimum(1.0, epoch / 10.)
                self.grad_loss = self.loss_mse + regr * self.loss_kl
            else:
                self.grad_loss = self.loss_mse + self.kl_c * self.loss_kl

        if return_output == 'hard':
            # return max(logits)
            return self.predict(logits)
        elif return_output == 'soft':
            # return softmax(logits)
            return self.softmax(logits)
        elif return_output == 'alpha':
            return alpha
        else:
            raise AssertionError

    def compute_mse(self, labels_1hot, evidence):
        num_classes = evidence.shape[-1]

        gap = labels_1hot - (evidence + self.lamb2) / \
              (evidence + self.lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + self.lamb2 * num_classes)

        loss_mse = gap.pow(2).sum(-1)

        return loss_mse.mean()

    def compute_kl_loss(self, alphas, target_concentration, epsilon=1e-8):
        target_alphas = torch.ones_like(alphas) * target_concentration

        alp0 = torch.sum(alphas, dim=-1, keepdim=True)
        target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

        alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
        alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
        assert torch.all(torch.isfinite(alp0_term)).item()

        alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                                + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                              torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
        alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
        assert torch.all(torch.isfinite(alphas_term)).item()

        loss = torch.squeeze(alp0_term + alphas_term).mean()

        return loss

    def step(self):
        self.optimizer.zero_grad()
        self.grad_loss.backward()
        self.optimizer.step()

    def predict(self, p):
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred