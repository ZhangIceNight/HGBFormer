import torch
import torch.nn as nn
from SoftPool import soft_pool2d, SoftPool2d
from adaPool import adapool2d, AdaPool2d


class DPM(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

    self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=3, dilation=2, stride=1, padding=2)
    self.conv2 = nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1, padding=0)
    
    self.bn1 = nn.BatchNorm2d(dim//2)
    self.bn2 = nn.BatchNorm2d(dim)
  

    self.softpool = SoftPool2d(kernel_size=(1,None))
    self.adapool = AdaPool2d(beta=(1,1), kernel_size=(None,1))
    self.gelu = nn.GELU()
  def forward(self, x):
    x = self.gelu(self.bn1(self.conv1(x)))
    x1 = self.softpool(x)
    x2 = self.adapool(x)
    x = torch.matmul(x1,x2)
    x = self.gelu(self.bn2(self.conv2(x)))
    return x


if __name__ == "__main__":
  b, h, w, c = 4, 224, 224, 48
  x = torch.randn([b,c,h,w]).cuda()
  dpm = DPM(dim=c).cuda()
  y = dpm(x)
  
  print(y.shape)
