import torch
import torch.nn as nn
from layer.ResidualBlock import ResidualBlock

class Encoder(nn.Module):
    
    def __init__(self,
                n_blocks=8,
                n_levels=4,
                input_ch=3,
                z_dim=10,
                UseMultiResSkips=True):
        
        super(Encoder, self).__init__()
        
        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.UseMultiResSkips = UseMultiResSkips
        
        self.conv_list = nn.ModuleList()
        self.res_blk_list = nn.ModuleList()
        self.multi_res_skip_list = nn.ModuleList()
        
        #For scf encoder change the input_conv layer channels 12 --> 8????
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_ch, out_channels=8,
                     kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        for i in range(n_levels):
            n_filters_1 = 2 ** (i+3)
            n_filters_2 = 2 ** (i+4)
            ks = 2 ** (n_levels - i)
            
            self.res_blk_list.append(
                nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1) 
                                for _ in range(n_blocks)])
            )
            
            self.conv_list.append(
                    torch.nn.Sequential(
                    torch.nn.Conv2d(n_filters_1, n_filters_2,
                                    kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_2),
                    torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if UseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters,
                                    kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                    torch.nn.BatchNorm2d(self.max_filters),
                    torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
                

        self.output_conv = torch.nn.Conv2d(in_channels=self.max_filters, out_channels=z_dim,
                                           kernel_size=(3, 3), stride=(1, 1), padding=1)
    
    
    def forward(self, x):

        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.UseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.UseMultiResSkips:
            x = sum([x] + skips)

        x = self.output_conv(x)

        return x
    
