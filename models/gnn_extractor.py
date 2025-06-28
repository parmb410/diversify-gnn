import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, lstm_hidden, output_dim):
        super(TemporalGCN, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for gcn in self.gcn_layers:
            x = F.relu(gcn(x, edge_index))
        x_pooled = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        x_pooled = x_pooled.unsqueeze(1)       # [batch, seq, feat]
        _, (hn, _) = self.lstm(x_pooled)
        hn = hn[-1]
        out = self.out(hn)
        return out

def build_correlation_graph(batch_time_series):
    # batch_time_series: [batch, channels, timesteps]
    # Returns a list of torch_geometric.data.Data objects
    from torch_geometric.data import Data
    data_list = []
    batch, channels, timesteps = batch_time_series.shape
    for i in range(batch):
        x = batch_time_series[i].T  # [timesteps, channels]
        x = x.float()
        corr = torch.corrcoef(x.T)
        edge_index = torch.nonzero(corr > 0.2, as_tuple=False).T
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
    return data_list
