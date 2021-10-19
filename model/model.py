import torch
import torch.nn as nn
from model.Encoder import Encoder
from model.Decoder import Decoder


class ResNetAE(nn.Module):
    
    def __init__(self,
                 inp_shape=[512, 512, 18],
                 n_ResidualBlock=3,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 UseMultiResSkips=True):
        
        super(ResNetAE, self).__init__()

        #assert input_shape[0] == input_shape[1]

        self.z_dim = z_dim
        self.img_latent_dim = inp_shape[0] // (2 ** n_levels)

        self.encoder_rgb_curr = Encoder(n_blocks=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=3, z_dim=z_dim, UseMultiResSkips=UseMultiResSkips)
        
        self.encoder_rgb_next = Encoder(n_blocks=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=3, z_dim=z_dim, UseMultiResSkips=UseMultiResSkips)
        
        self.encoder_scf = Encoder(n_blocks=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=12, z_dim=z_dim, UseMultiResSkips=UseMultiResSkips)
        
        
        self.decoder = Decoder(n_blocks=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=12, z_dim=z_dim, UseMultiResSkips=UseMultiResSkips)
        
        dim = int(self.z_dim)*int(self.img_latent_dim)*int(self.img_latent_dim)
        #self.fc1_rgb1 = nn.Linear(dim, bottleneck_dim)
        #self.fc1_rgb2 = nn.Linear(dim, bottleneck_dim)
        #self.fc1_scf = nn.Linear(dim, bottleneck_dim)
        #self.fc2 = nn.Linear(bottleneck_dim, dim)
        
        
    def encode_rgb1(self, x):
    
        h = self.encoder_rgb_curr(x)

        return h #self.fc1_rgb1(h.reshape(-1, self.z_dim*self.img_latent_dim*self.img_latent_dim))
    
    def encode_rgb2(self, x):
    
        h = self.encoder_rgb_next(x) #B,z_dim,32,32

        #h= self.fc1_rgb2(h.reshape(-1, self.z_dim*self.img_latent_dim*self.img_latent_dim))
        return h
    
    def encode_scf(self, x):
    
        h = self.encoder_scf(x)

        #h = self.fc1_scf(h.reshape(-1, self.z_dim*self.img_latent_dim*self.img_latent_dim))
        return h

    def decode(self, z):
        h = self.decoder(z)

        #h = self.decoder(self.fc2(z)).reshape(-1, self.z_dim*3, self.img_latent_dim, self.img_latent_dim))
        #return torch.sigmoid(h)
        return h
        

    def forward(self, x):
        
        #x = x.permute(0, 3, 1, 2)#B*KxCxHxW

        #z_rgb1_embed = self.encode_rgb1(x[:,:3,:,:])
        #z_rgb2_embed = self.encode_rgb2(x[:,3:6,:,:])
        #z_scf_embed = self.encode_scf(x[:,6:,:,:])
        
        z_concat = self.decode(torch.cat([self.encode_rgb1(x[:,:3,:,:]) ,
                              self.encode_rgb2(x[:,3:6,:,:]) ,
                              self.encode_scf(x[:,6:,:,:])], dim=1)) #B,z_dimx3,32,32
        
        #z_concat = z_concat.to(torch.float16)
        #del z_rgb1_embed, z_rgb2_embed, z_scf_embed

        out = (z_concat)

        return out
    
    
class ResNetAE_RGB(nn.Module):
    
    def __init__(self,
                 inp_shape=[512, 512, 27],
                 n_ResidualBlock=4,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 UseMultiResSkips=True):
        
        super(ResNetAE_RGB, self).__init__()

        #assert input_shape[0] == input_shape[1]

        self.z_dim = z_dim
        self.img_latent_dim = inp_shape[0] // (2 ** n_levels)

        self.encoder_rgb_curr = Encoder(n_blocks=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=12, z_dim=z_dim, UseMultiResSkips=UseMultiResSkips)
        
        self.encoder_rgb_next = Encoder(n_blocks=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=3, z_dim=z_dim, UseMultiResSkips=UseMultiResSkips)
        
        self.encoder_scf = Encoder(n_blocks=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=12, z_dim=z_dim, UseMultiResSkips=UseMultiResSkips)
        
        
        self.decoder = Decoder(n_blocks=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=12, z_dim=z_dim, UseMultiResSkips=UseMultiResSkips)
        
        dim = int(self.z_dim)*int(self.img_latent_dim)*int(self.img_latent_dim)
        #self.fc1_rgb1 = nn.Linear(dim, bottleneck_dim)
        #self.fc1_rgb2 = nn.Linear(dim, bottleneck_dim)
        #self.fc1_scf = nn.Linear(dim, bottleneck_dim)
        #self.fc2 = nn.Linear(bottleneck_dim, dim)
        
        
    def encode_rgb1(self, x):
    
        h = self.encoder_rgb_curr(x)

        return h #self.fc1_rgb1(h.reshape(-1, self.z_dim*self.img_latent_dim*self.img_latent_dim))
    
    def encode_rgb2(self, x):
    
        h = self.encoder_rgb_next(x) #B,z_dim,32,32

        #h= self.fc1_rgb2(h.reshape(-1, self.z_dim*self.img_latent_dim*self.img_latent_dim))
        return h
    
    def encode_scf(self, x):
    
        h = self.encoder_scf(x)

        #h = self.fc1_scf(h.reshape(-1, self.z_dim*self.img_latent_dim*self.img_latent_dim))
        return h

    def decode(self, z):
        h = self.decoder(z)

        #h = self.decoder(self.fc2(z)).reshape(-1, self.z_dim*3, self.img_latent_dim, self.img_latent_dim))
        #return torch.sigmoid(h)
        return h
        

    def forward(self, x):
        
        #x = x.permute(0, 3, 1, 2)#B*KxCxHxW

        #z_rgb1_embed = self.encode_rgb1(x[:,:3,:,:])
        #z_rgb2_embed = self.encode_rgb2(x[:,3:6,:,:])
        #z_scf_embed = self.encode_scf(x[:,6:,:,:])
        
        z_concat = self.decode(torch.cat([self.encode_rgb1(x[:,:12,:,:]) ,
                              self.encode_rgb2(x[:,12:15,:,:]) ,
                              self.encode_scf(x[:,15:,:,:])], dim=1)) #B,z_dimx3,32,32
        
        #z_concat = z_concat.to(torch.float16)
        #del z_rgb1_embed, z_rgb2_embed, z_scf_embed

        out = (z_concat)

        return out
