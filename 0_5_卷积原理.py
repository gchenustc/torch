import torch
import torch.nn.functional as F

# torch.nn.functional.conv2d

# 创建数据 - 注意这里是float32否则F.conv2d会报错
input = torch.tensor([[1,2,3,4],
                      [0,1,2,3],
                      [0,1,0,1],
                      [1,0,1,0]], dtype=torch.float32)

kernel = torch.ones((3,3))

input = input.view((1,1,4,4))  # batch, channel, H, W
kernel = kernel.view((1,3,3))

ret = F.conv2d(input, kernel, bias=0, stride=1, padding=0)
print(ret)