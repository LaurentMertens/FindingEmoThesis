"""
EmoNet + additional FC layer to perform regression and predict arousal.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import torch
from emonet import EmoNet


class EmoNetArousal(torch.nn.Module):
    FC_BIAS = 6.5139
    FC_WEIGHTS = [-0.0910, -0.0121, -0.3154, 0.0152, 0.3949, -0.2019, -0.1216, -0.1029, 0.0637, 0.0037,
                  -0.1700, 0.0736, 0.2145, 0.1670, -0.2425, -0.0555, 0.1480, 0.1390, 0.1877, -0.0824]

    def __init__(self):
        super().__init__()
        self.emonet = EmoNet.get_emonet(b_eval=False)
        self.fc = torch.nn.Linear(in_features=20, out_features=1, bias=True)
        self.fc.bias = torch.nn.Parameter(torch.as_tensor(self.FC_BIAS))
        self.fc.weight = torch.nn.Parameter(torch.as_tensor(self.FC_WEIGHTS).unsqueeze(0))

    def forward(self, x):
        y = self.emonet(x)
        res = self.fc(y)

        return res
