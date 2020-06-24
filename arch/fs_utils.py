import torch.nn as nn

class Model_FS(nn.Module):
    def __init__(self, basic_net):
        super(Model_FS, self).__init__()
        self.basic_net = basic_net
        self.basic_net.eval()

    def forward(self, inputs):      
        outputs, _ = self.basic_net(inputs)
        return outputs