"""Model definitions: MaskedMAELoss, LSTM baseline, CNN-LSTM and kernel-size variant."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMAELoss(nn.Module):
    """Compute MAE only over the inhalation interval (u_out == 0)."""

    def forward(self, pred, target, u_out_mask):
        valid = u_out_mask == 0
        abs_err = torch.abs(pred - target)
        err = abs_err[valid]
        if err.numel() == 0:
            return (pred * 0.0).sum()
        return torch.mean(err)


class LstmBaseline(nn.Module):
    """Single-layer unidirectional LSTM + Linear, predicting pressure per time step."""

    def __init__(self, input_feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_feat_dim, hidden_size=hidden_dim,
            num_layers=1, bidirectional=False, batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out).squeeze(-1)


class CnnLstm(nn.Module):
    """Conv1d + ReLU local temporal features followed by an LSTM."""

    def __init__(self, input_feat_dim: int, hidden_dim: int = 64, kernel_size: int = 3,
                 out_channel: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_feat_dim, out_channel, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2,
        )
        self.lstm = nn.LSTM(out_channel, hidden_dim, num_layers=1,
                            bidirectional=False, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.linear(out).squeeze(-1)
