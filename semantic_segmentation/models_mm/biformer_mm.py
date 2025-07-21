
import torch
import torch.nn as nn
from mmcv_custom import load_checkpoint
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from .biformer import BiFormer 
from timm.models.layers import LayerNorm2d
from .DPM import DPM
from .FAMPatchMerging import FAMPatchMerging
from .HyperGraphBlock import HGNNPBlock

@BACKBONES.register_module()  
class BiFormer_mm(BiFormer):
    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)
        
        # step 1: remove unused segmentation head & norm
        del self.head # classification head
        del self.norm # head norm
        
        #del self.downsample_layers
        
        # step 2: add extra norms for dense tasks
        self.extra_norms = nn.ModuleList()
        
        # step 3: add HGMs DPMs and FAMs
        self.hg_layers = nn.ModuleList()
        self.dpms = nn.ModuleList()
        self.dpmPatchMergings = nn.ModuleList()
        
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
            self.dpms.append(DPM(self.embed_dim[i]))
            self.FAMPatchMergings.append(FAMPatchMerging(self.embed_dim[i]//2))
            self.hg_layers.append(HGNNPBlock(self.embed_dim[i], self.embed_dim[i]))
        
        self.preEmb = nn.Sequential(
            nn.Conv2d(3, self.embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.embed_dim[0] // 2),
            nn.GELU()
        ) 



        # step 4: initialization & load ckpt
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)

        # step 5: convert sync bn, as the batch size is too small in segmentation
        nn.SyncBatchNorm.convert_sync_batchnorm(self)


    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            print(f'Load pretrained model from {pretrained}')   
    
    def forward_features(self, x: torch.Tensor):
        out = []
        y = self.preEmb(x)
        for i in range(4):
            if i==0:
                y = self.FAMPatchMergings[i](y)
            else:
                y = self.FAMPatchMergings[i](x)
            x = self.downsample_layers[i](x)
            x = x + y
            
            x = self.stages[i](x)
            short = self.dpms[i](x)
            short = self.hg_layers[i](short)
            x = x + short

            del y
            del short
            out.append(self.extra_norms[i](x))

        return tuple(out)
    
    def forward(self, x:torch.Tensor):
        return self.forward_features(x)


if __name__ == "__main__":
  b, h, w, c = 4, 224, 224, 48
  x = torch.randn([b,c,h,w]).cuda()

