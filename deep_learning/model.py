import torch.nn as nn
import torch
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.in_conv = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) 
        
        # Decoder
        x = self.up1(x5)  
        diffY = x4.size()[2] - x.size()[2]
        diffX = x4.size()[3] - x.size()[3]
        
        #these sequenes handle mismatch dimensions
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        elif diffY < 0 or diffX < 0:
            x4 = F.pad(x4, [-diffX // 2, -(diffX - diffX // 2), -diffY // 2, -(diffY - diffY // 2)])
        x = self.conv1(torch.cat([x, x4], dim=1)) 
        
        x = self.up2(x)  
        # Handle dimension mismatch
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        elif diffY < 0 or diffX < 0:
            x3 = F.pad(x3, [-diffX // 2, -(diffX - diffX // 2), -diffY // 2, -(diffY - diffY // 2)])
        x = self.conv2(torch.cat([x, x3], dim=1))
        
        x = self.up3(x)
        # Handle dimension mismatch
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        elif diffY < 0 or diffX < 0:
            x2 = F.pad(x2, [-diffX // 2, -(diffX - diffX // 2), -diffY // 2, -(diffY - diffY // 2)])
        x = self.conv3(torch.cat([x, x2], dim=1))
        
        x = self.up4(x)
        # Handle dimension mismatch
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        elif diffY < 0 or diffX < 0:
            x1 = F.pad(x1, [-diffX // 2, -(diffX - diffX // 2), -diffY // 2, -(diffY - diffY // 2)])
        x = self.conv4(torch.cat([x, x1], dim=1))
        
        # Output layer
        return self.out_conv(x)
    
    
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU => Dropout) * 2"""
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        # Add dropout if specified
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
            
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        
        # Add dropout after second conv block too
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
            
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)    
    
# Alternative: Simple uniform dropout version
class UNetSimpleDropout(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_p=0.5):
        super(UNetSimpleDropout, self).__init__()

        # Same dropout rate throughout
        self.in_conv = DoubleConv(n_channels, 64, dropout_p=dropout_p)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128, dropout_p=dropout_p)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256, dropout_p=dropout_p)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512, dropout_p=dropout_p)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024, dropout_p=dropout_p)
        )

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512, dropout_p=dropout_p)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256, dropout_p=dropout_p)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128, dropout_p=dropout_p)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64, dropout_p=dropout_p)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # REMOVED: 1024-channel bottleneck layer
        
        # Decoder with dimension mismatch handling
        x = self.up1(x5)  # REMOVED: First upsampling from 1024 to 512
        # Handle dimension mismatch 
        diffY = x4.size()[2] - x.size()[2]
        diffX = x4.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        elif diffY < 0 or diffX < 0:
            x4 = F.pad(x4, [-diffX // 2, -(diffX - diffX // 2), -diffY // 2, -(diffY - diffY // 2)])
        x = self.conv1(torch.cat([x, x4], dim=1))  # REMOVED: First decoder conv block
        
        # Start decoder directly from x4 (512 channels is now the bottleneck)
        x = self.up2(x)  # CHANGED: Now starts from x4 instead of previous x
        # Handle dimension mismatch
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        elif diffY < 0 or diffX < 0:
            x3 = F.pad(x3, [-diffX // 2, -(diffX - diffX // 2), -diffY // 2, -(diffY - diffY // 2)])
        x = self.conv2(torch.cat([x, x3], dim=1))
        
        x = self.up3(x)
        # Handle dimension mismatch
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        elif diffY < 0 or diffX < 0:
            x2 = F.pad(x2, [-diffX // 2, -(diffX - diffX // 2), -diffY // 2, -(diffY - diffY // 2)])
        x = self.conv3(torch.cat([x, x2], dim=1))
        
        x = self.up4(x)
        # Handle dimension mismatch
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        elif diffY < 0 or diffX < 0:
            x1 = F.pad(x1, [-diffX // 2, -(diffX - diffX // 2), -diffY // 2, -(diffY - diffY // 2)])
        x = self.conv4(torch.cat([x, x1], dim=1))
        
        # Output layer
        return self.out_conv(x)