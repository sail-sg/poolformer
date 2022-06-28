"""
An example to use fvcore to count MACs.

please install the following packages
`pip install timm fvcore`
"""
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import timm

if __name__ == '__main__':
    model = timm.models.poolformer_s12()
    model.eval()
    image_size = [224, 224]
    input = torch.rand(1, 3, *image_size)

    # Please note that FLOP here actually means MAC.
    flop = FlopCountAnalysis(model, input)
    # print(flop_count_table(flop, max_depth=4))
    print('MACs (G):', flop.total()/1e9)