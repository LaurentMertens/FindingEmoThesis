"""
EmoNet + additional FC layer to perform regression and predict valence.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import torch

from emonet import EmoNet


class EmoNetValence(torch.nn.Module):
    FC_BIAS = 3.2963
    FC_WEIGHTS = [0.5990, 0.4576, 0.3248, -0.0634, 0.0035, 0.0560, 0.1163, -0.1370, -0.7719, -0.1210,
                  -0.1862, 0.0214, -0.3015, -0.3680, 0.2467, -0.2335, 0.1096, -0.3858, 0.1538, 0.5741]

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
