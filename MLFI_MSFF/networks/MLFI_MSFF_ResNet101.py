import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import numpy as np
affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out
        
        
class MSFF_Residual_Connection(nn.Module):

    def __init__(self, inplanes, planes, dilation=1):
        super(MSFF_Residual_Connection, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=True) # change
        
        self.conv1.weight.data.normal_(0, 0.01)
        
        self.bn1 = nn.BatchNorm2d(inplanes,affine = affine_par)
        """
        for i in self.bn1.parameters():
            i.requires_grad = False
            """

        padding = dilation
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, # change
                               padding=padding, bias=True, dilation = dilation)
        
        self.conv2.weight.data.normal_(0, 0.01)
        
        self.bn2 = nn.BatchNorm2d(inplanes,affine = affine_par)
        """
        for i in self.bn2.parameters():
            i.requires_grad = False
            """
        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=True)
        
        self.conv3.weight.data.normal_(0, 0.01)
        
        self.bn3 = nn.BatchNorm2d(inplanes, affine = affine_par)
        
        
        self.droplayer = nn.Dropout2d(0.5)
        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=True)
        """
        for i in self.bn3.parameters():
            i.requires_grad = False
            """
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out = out + residual
        
        out = self.droplayer(out)
        out = self.conv4(out)
        out = self.relu(out)
        

        return out
        
        
    
class MLFI_Residual_Connection(nn.Module):

    def __init__(self, inplanes, planes, dilation):
        super(MLFI_Residual_Connection, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=True) # change
        
        self.conv1.weight.data.normal_(0, 0.01)
        
        self.bn1 = nn.BatchNorm2d(inplanes,affine = affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, # change
                               padding=padding, bias=True, dilation = dilation)
        
        self.conv2.weight.data.normal_(0, 0.01)
        
        self.bn2 = nn.BatchNorm2d(inplanes,affine = affine_par)


        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=True)
        
        self.conv3.weight.data.normal_(0, 0.01)
        
        self.bn3 = nn.BatchNorm2d(inplanes, affine = affine_par)
        
        
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        

        out = out + residual
        
        out = self.relu(out)
        
        return out
    

    
class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(MSFF_Residual_Connection(256, num_classes, dilation=dilation))
        

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out = out + self.conv2d_list[i+1](x)
        return out

"""
class Residual_Covolution(nn.Module):
    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg
        """
    

class MLFI_MSFF_NET(nn.Module):
    def __init__(self, block, num_blocks,num_classes):
        
        self.inplanes = 64
        super(MLFI_MSFF_NET, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.smooth_block1 = self._make_smooth_block(MLFI_Residual_Connection, 256, 256, dilation=2)
        self.smooth_block2 = self._make_smooth_block(MLFI_Residual_Connection, 256, 256, dilation=2)
        self.smooth_block3 = self._make_smooth_block(MLFI_Residual_Connection, 256, 256, dilation=2)
        
        
        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    
    def _make_smooth_block(self, block, inplanes, planes, dilation):
        return block(inplanes, planes, dilation)
        
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        # c1 = F.relu(self.bn1(self.conv1(x)))
        # c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        # print c2.size()
        c3 = self.layer2(c2)
        # print c3.size()
        c4 = self.layer3(c3)
        # print c4.size()
        c5 = self.layer4(c4)
        # print c5.size()
        # Top-down
        p5 = self.latlayer1(c5)
        # print p5.size()
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.smooth_block1(p4)
        # print p4.size()
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.smooth_block2(p3)
        # print p3.size()
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.smooth_block3(p2)
        # print p2.size()
        p1 = self._upsample_add(self._upsample_add(p4, p3), p2)
        out = self.layer5(p1)
        # print out.size()
        return out# , p2, p3, p4, p5


def MLFI_MSFF_ResNet(num_classes=2):
    model = MLFI_MSFF_NET(Bottleneck, [3,4,23,3], num_classes)
    return model
