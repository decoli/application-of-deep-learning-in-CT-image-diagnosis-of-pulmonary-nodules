'''
coloring the attention area by VAE model
'''

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        # input 1* 50 * 50

        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=20, kernel_size=7, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        ) # 20 * 44 * 44

        self.pooling_1 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False
        ) # 20 * 22 * 22

        self.conv_2 = nn.Conv2d(
            in_channels=20, out_channels=50, kernel_size=7, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        ) # 50 * 16 * 16

        self.pooling_2 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False
        ) # 50 * 8 * 8

        self.conv_3 = nn.Conv2d(
            in_channels=50, out_channels=500, kernel_size=7, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        ) # 500 * 2 * 2

        self.pooling_3 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False
        ) # 500 * 1 * 1

        # activate fuction ReLU layer

        self.conv_4 = nn.Conv2d(
            in_channels=500, out_channels=2, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        ) # 2 * 1 * 1

    def forward(self, input_image):
        out = self.conv_1(input_image) # 20 * 44 * 44
        out = self.pooling_1(out) # 20 * 22 * 22

        out = self.conv_2(out) # 50 * 16 * 16
        out = self.pooling_2(out) # 50 * 8 * 8

        out = self.conv_3(out) # 500 * 2 * 2
        out = self.pooling_3(out) # 500 * 1 * 1

        out_relu = F.relu(out)

        # out = F.dropout(out_relu)
        out = self.conv_4(out) # 2 * 1 * 1

        return out, out_relu
