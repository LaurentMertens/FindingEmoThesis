"""
A small script to verify that the output of the PyTorch EmoNet equals that of the original MatLab version.
For this, we use two demo images, situated in the 'data' folder.
'demo_big.jpg' is a 800x450 photo, 'demo_small.jpg' is a center 450x450 center crop of the big photo, resized to 227x227.

The demo picture has been taken by the author, i.e., Laurent Mertens, so there is no copyright infringement for sure.

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import unittest
import os

import torch

from emonet_py import EmoNet


class TestIntegrity(unittest.TestCase):
    def test_integrity(self):
        img_big = os.path.join('..', 'data', 'demo_big.jpg')
        img_small = os.path.join('..', 'data', 'demo_small.jpg')

        # These are the reference output values for both images, as copy/pasted from the
        # MatLab output.
        ref_big = [0.0044732, 0.0945, 0.034106, 0.3361, 0.021589,
                   0.0090503, 0.00067812, 0.05514, 0.072519, 0.019064,
                   0.0032458, 0.017255, 0.030013, 0.00058266, 0.20558,
                   0.0099342, 0.059141, 0.00026909, 4.2833e-05, 0.026715]
        ref_big = torch.tensor(ref_big, dtype=torch.float32)
        ref_small = [0.22926, 0.075324, 0.0030083, 0.0024761, 0.51978,
                     0.0075244, 0.0028647, 0.0041199, 0.0040441, 0.021392,
                     0.00352, 0.0068833, 0.015457, 0.0014801, 0.070372,
                     0.0075813, 0.011797, 6.71e-05, 4.6644e-06, 0.013039]
        ref_small = torch.tensor(ref_small, dtype=torch.float32)

        emonet = EmoNet(b_eval=True)

        pred_big = emonet.emonet(emonet.load_and_preprocess_image(img_big).unsqueeze(0)).squeeze(0)
        # Check all MatLab vs. EmoNet probability comparisons lie within a relative tolerance of 1%
        # of each other.
        check_big = torch.isclose(ref_big, pred_big, rtol=1e-2)
        self.assertEqual(sum(check_big), 20)

        pred_small = emonet.emonet(emonet.load_and_preprocess_image(img_small).unsqueeze(0)).squeeze(0)
        # Check all MatLab vs. EmoNet probability comparisons lie within a relative tolerance of .1%
        # of each other.
        # Since this image does not need to be rescaled, a process that produces slightly different results
        # in MatLab and Python, we can set the relative tolerance a bit tighter.
        check_small = torch.isclose(ref_small, pred_small, rtol=1e-3)
        self.assertEqual(sum(check_small), 20)
