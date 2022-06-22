import os

import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from pn2_regressor import Net
from pointcloud_dataset import PointCloudsInFiles

def write_las(outpoints, outfilepath, attribute_dict={}):
    """
    :param outpoints: 3D array of points to be written to output file
    :param outfilepath: specification of output file (format: las or laz)
    :param attribute_dict: dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    :return: None
    """
    import laspy
    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.x_scale = 0.00025
    hdr.y_scale = 0.00025
    hdr.z_scale = 0.00025
    mean_extent = np.mean(outpoints, axis=0)
    hdr.x_offset = int(mean_extent[0])
    hdr.y_offset = int(mean_extent[1])
    hdr.z_offset = int(mean_extent[2])

    las = laspy.LasData(hdr)

    las.x = outpoints[:, 0]
    las.y = outpoints[:, 1]
    las.z = outpoints[:, 2]
    for key, vals in attribute_dict.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name=key,
                type=type(vals[0])
            ))
            las[key] = vals

    las.write(outfilepath)


if __name__ == '__main__':

    use_columns = ['scan_angle_rank']
    train_dataset = PointCloudsInFiles(r'C:\Users\hseely\OneDrive - UBC\Documents\Jupyter_Lab_Workspace\Lukas_DL_Regression_Example\train',
                                       '*.laz',
                                       'NormalizedZ', max_points=4_000, use_columns=use_columns)
    test_dataset = PointCloudsInFiles(r'C:\Users\hseely\OneDrive - UBC\Documents\Jupyter_Lab_Workspace\Lukas_DL_Regression_Example\test',
                                      '*.laz',
                                      'NormalizedZ', max_points=4_000, use_columns=use_columns)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")
    model = Net(num_features=len(use_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    def train():
        model.train()

        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print(f'[{i + 1}/{len(train_loader)}] MSE Loss: {loss.to("cpu"):.4f} ')


    @torch.no_grad()
    def test(loader, ep_id):
        model.eval()
        losses = []
        for idx, data in enumerate(loader):
            data = data.to(device)
            outs = model(data)
            loss = F.mse_loss(outs, data.y)
            losses.append(float(loss.to("cpu")))
            if idx == 0:
                batch = data.batch.to('cpu').numpy()
                coords = data.pos.to('cpu').numpy()[batch==0, :]
                vals = data.y.to('cpu').numpy()[batch==0, 0]
                vals_pred = outs.to('cpu').numpy()[batch==0, 0]
                write_las(coords, rf'C:\Users\hseely\OneDrive - UBC\Documents\Jupyter_Lab_Workspace\Lukas_DL_Regression_Example\predicted\ep{ep_id}_{idx}.laz',
                          {'ref': vals,
                           'pred': vals_pred
                           } )
        return float(np.mean(losses))


    for epoch in range(1, 501):
        model_path = rf'C:\Users\hseely\OneDrive - UBC\Documents\Jupyter_Lab_Workspace\Lukas_DL_Regression_Example\predicted\latest.model'
        if os.path.exists(model_path):
            model = torch.load(model_path)
        train()
        mse = test(test_loader, epoch)
        torch.save(model, model_path)
        print(f'Epoch: {epoch:02d}, Mean test MSE: {mse:.4f}')
