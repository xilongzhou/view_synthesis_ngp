import torch
import torch.nn as nn
import torch.nn.functional as F

class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """


        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
           
    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.
        Parameters
        ----------
            x : tensor
                input to the NN block.
        Returns
        -------
            tensor
                output of the NN block.
        """


        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x
    
class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.
    ...
    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        
        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
           
    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.
        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.
        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        # # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # x = F.sigmoid(self.conv1(x))

        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope = 0.1)
        return x



class Mask(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.
    
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """


    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        
        super(Mask, self).__init__()
        # Initialize neural network blocks.
        nfg = 2
        self.conv1 = nn.Conv2d(inChannels, nfg*8, 7, stride=1, padding=3)
        # self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(nfg*8, nfg*16, 5)
        self.down2 = down(nfg*16, nfg*32, 3)
        self.down3 = down(nfg*32, nfg*64, 3) # comment for D2
        self.down4 = down(nfg*64, nfg*128, 3) # comment for D2
        self.down5 = down(nfg*128, nfg*128, 3) # comment for D2
        self.up1   = up(nfg*128, nfg*128) # comment for D2
        self.up2   = up(nfg*128, nfg*64) # comment for D2
        self.up3   = up(nfg*64, nfg*32) # comment for D2
        self.up4   = up(nfg*32, nfg*16)
        self.up5   = up(nfg*16, nfg*8)
        self.conv3 = nn.Conv2d(nfg*8, outChannels, 3, stride=1, padding=1)
        
    def forward(self, x):#, flow21, image1, warped1):
#         """
#         Returns output tensor after passing input `x` to the neural network
#         block.
#         Parameters
#         ----------
#             x : tensor
#                 input to the NN block.
#         Returns
#         -------
#             tensor
#                 output of the NN block.
#         """

        # x = torch.cat((flow21, image1, warped1), 1)


        s1  = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # s1 = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2) # x for D2 s3 for D5
        s4 = self.down3(s3) # comment for D2
        s5 = self.down4(s4) # comment for D2
        x  = self.down5(s5) # comment for D2
        x  = self.up1(x, s5) # comment for D2
        x  = self.up2(x, s4) # comment for D2
        x  = self.up3(x, s3) # comment for D2
        x  = self.up4(x, s2)
        x  = self.up5(x, s1)
        x  = torch.sigmoid(self.conv3(x))#-0.5
        return x