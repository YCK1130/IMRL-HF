import torch
import torch.nn as nn
import torch.nn.functional as F

from .util.config import DEVICE
from .util.torch_utils import tensor


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x


class SingleOptionNet(nn.Module):
    def __init__(self,
                 action_dim,
                 body_fn):
        super(SingleOptionNet, self).__init__()

        self.output_keys = ['mean', 'std', 'beta']

        self.pi_body = body_fn()
        self.beta_body = body_fn()
        self.fc_pi = layer_init(
            nn.Linear(self.pi_body.feature_dim, action_dim), 1e-3)
        self.fc_beta = layer_init(
            nn.Linear(self.beta_body.feature_dim, 1), 1e-3)
        self.std = nn.Parameter(torch.zeros((1, action_dim)))

    def forward(self, phi):
        phi_pi = self.pi_body(phi)
        mean = F.tanh(self.fc_pi(phi_pi))
        std = F.softplus(self.std).expand(mean.size(0), -1)

        phi_beta = self.beta_body(phi)
        beta = F.sigmoid(self.fc_beta(phi_beta))

        return {
            'mean': mean,
            'std': std,
            'beta': beta,
        }


class OptionGaussianActorCriticNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 num_options,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 option_body_fn=None):
        super(OptionGaussianActorCriticNet, self).__init__()

        self.output_keys = ['mean', 'std', 'q_o',
                            'u_o', 'inter_pi', 'log_inter_pi', 'beta']

        if phi_body is None:
            phi_body = DummyBody(state_dim)
        if critic_body is None:
            critic_body = DummyBody(phi_body.feature_dim)
        if actor_body is None:
            actor_body = DummyBody(phi_body.feature_dim)

        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body

        self.options = nn.ModuleList(
            [SingleOptionNet(action_dim, option_body_fn) for _ in range(num_options)])

        self.fc_pi_o = layer_init(
            nn.Linear(actor_body.feature_dim, num_options), 1e-3)
        self.fc_q_o = layer_init(
            nn.Linear(critic_body.feature_dim, num_options), 1e-3)
        self.fc_u_o = layer_init(
            nn.Linear(critic_body.feature_dim, num_options + 1), 1e-3)

        self.num_options = num_options
        self.action_dim = action_dim
        self.to(DEVICE)

    def forward(self, obs):
        obs = tensor(obs)
        phi = self.phi_body(obs)

        mean = []
        std = []
        beta = []
        for option in self.options:
            prediction = option(phi)
            mean.append(prediction['mean'].unsqueeze(1))
            std.append(prediction['std'].unsqueeze(1))
            beta.append(prediction['beta'])

        mean = torch.cat(mean, dim=1)
        std = torch.cat(std, dim=1)
        beta = torch.cat(beta, dim=1)

        phi_a = self.actor_body(phi)
        phi_a = self.fc_pi_o(phi_a)
        pi_o = F.softmax(phi_a, dim=-1)
        log_pi_o = F.log_softmax(phi_a, dim=-1)

        phi_c = self.critic_body(phi)
        q_o = self.fc_q_o(phi_c)
        u_o = self.fc_u_o(phi_c)

        return {
            'mean': mean,
            'std': std,
            'q_o': q_o,
            'u_o': u_o,
            'inter_pi': pi_o,
            'log_inter_pi': log_pi_o,
            'beta': beta
        }
