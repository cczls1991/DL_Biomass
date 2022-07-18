import torch
from torch.nn import Linear
from torch_geometric.nn import MLP, knn_interpolate, PointConv, global_max_pool, fps, radius

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self, num_features, activation_function, neuron_multiplier, dropout_probability):
        super().__init__()

        if neuron_multiplier == 0:
            neuron_multiplier = 1  # Use original architecture when neuron multiplier is zero
        else:
            neuron_multiplier = neuron_multiplier

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 2, MLP([3 + num_features, 64*neuron_multiplier, 64*neuron_multiplier, 128*neuron_multiplier], act=activation_function))
        self.sa2_module = SAModule(0.25, 8, MLP([128*neuron_multiplier + 3, 128*neuron_multiplier, 128*neuron_multiplier, 256*neuron_multiplier], act=activation_function))
        self.sa3_module = GlobalSAModule(MLP([256*neuron_multiplier + 3, 256*neuron_multiplier, 512*neuron_multiplier, 1024*neuron_multiplier], act=activation_function))

        self.mlp = MLP([1024*neuron_multiplier, 128*neuron_multiplier, 128*neuron_multiplier, 1], act=None, dropout=0.5)  # No activation function for the output layer following Oehmcke

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x, _, _ = self.sa3_module(*sa2_out)

        return self.mlp(x)

