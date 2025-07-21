import torch
import torch.nn as nn
from SoftPool import soft_pool2d, SoftPool2d
from adaPool import adapool2d, AdaPool2d


class FAMPatchMerging(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

    self.dconv = nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, dilation=2, padding=2)

    self.conv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1, stride=1)
    self.conv2 = nn.Conv2d(dim, 2 * dim, kernel_size=1, stride=1)
    self.conv3 = nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, stride=2)
    
    self.bn2 = nn.BatchNorm2d(2 * dim)
  
    self.gn1 = nn.GroupNorm(dim // 2, dim)

    self.adapool = AdaPool2d(beta=(1,1), kernel_size=(2, 2), stride=(2, 2))
    self.gelu = nn.GELU()
  def forward(self, x):
    B,H,W,C = x.shape
    short = x.float()
    x = self.gelu(self.bn2(self.conv1(x)))
    x = self.bn2(self.dconv(x)) 
    x = self.gelu(self.bn2(self.conv3(x)))
    short = self.gn1(self.adapool(short))
    short = self.gelu(self.bn2(self.conv2(short)))
    x = x + short

    return x


if __name__ == "__main__":
  b, h, w, c = 4, 224, 224, 48
  x = torch.randn([b,c,h,w]).cuda()
  patchMerging = FAMPatchMerging(dim=c).cuda()
  y = patchMerging(x)
  
  print(y.shape)
