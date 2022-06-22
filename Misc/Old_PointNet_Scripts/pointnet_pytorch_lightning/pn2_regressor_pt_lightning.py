import torch
from torch_geometric.nn import MLP, knn_interpolate, PointConv, global_max_pool, fps, radius
import pytorch_lightning as pl
import torch.nn.functional as F



class SAModule(pl.LightningModule):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.register_buffer("sigma", torch.eye(3)) # Allow access to self.sigma anywhere in module
        self.ratio = ratio
        self.r = r  #r is the radius of a ball where points are being considered
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        edge_index = edge_index.type_as(x) # Tell pl to do GPU training
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(pl.LightningModule):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.register_buffer("sigma", torch.eye(3)) # Allow access to self.sigma anywhere in module

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = x.type_as(x) # Tell pl to do GPU training
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        batch = batch.type_as(x)
        return x, pos, batch


class Net(pl.LightningModule):
    def __init__(self, num_features):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 2, MLP([3 + num_features, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 8, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 128, 128, 1], dropout=0.5,
                       batch_norm=False)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x, _, _ = self.sa3_module(*sa2_out)

        return self.mlp(x)

    def training_step(self, batch, batch_index):
        x, y = batch
        output = self.forward(x)
        loss = F.mse_loss(output, y)
        self.log("train loss", loss)
        return loss

    #Define validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.mse_loss(output, y)
        self.log('mse_loss', loss)

    # Define optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

