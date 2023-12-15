'''
-*- coding: utf-8 -*-

LinkNet model.

Author: Rayykia
'''

import torch
from torch import Tensor
import torch.nn as nn

from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import Union
from typing import Tuple


class ConvBlock(nn.Module):
    '''
    input ->
    conv((3,3))
    batch normalization
    relu
    -> output
    '''
    def __init__(
            self,
            in_planes: int, 
            out_planes: int, 
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: Union[int, Tuple[int, int]] = 1, 
            padding: Union[int, Tuple[int, int]] = 1,
            groups: int = 1, 
            dilation: Union[int, Tuple[int, int]] = 1,
            bias: bool = False,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError("LinkNet only supports groups = 1.")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported.")

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            ),
            norm_layer(num_features=out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        x = self.conv_bn_relu(x)
        return x




class FullConvBlock(nn.Module):
        '''
        input ->
        full-convolution (defult kernel size: 3x3),
        Optional[batch normalization, relu].
        -> output
        '''
        def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: Union[int, Tuple[int, int]] = 2,
            padding: Union[int, Tuple[int, int]] = 1,
            output_padding: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            dilation: Union[int, Tuple[int, int]] = 1,
            bias: bool = True,
            norm_layer: Optional[Callable[..., nn.Module]] = None
        ) -> None:
            super().__init__()

            self.full_conv = nn.ConvTranspose2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation
            )

            if norm_layer is None:
                self.norm_layer = nn.BatchNorm2d(out_planes)
            else:
                self.norm_layer = norm_layer
            
        def forward(
                self,
                x: Tensor,
                activate: bool = True
        ) -> Tensor:
            x = self.full_conv(x)

            if activate:
                x = torch.relu(self.norm_layer(x))

            return x




class EncoderBlock(nn.Module):
    '''
    Model Structure:

    input ->
    conv((3x3), (m,n), /2)
    conv((3x3), (n,n))
    -> out1
    
    input ->
    conv((3,3), (m,n), /2)
    -> downsampled_input

    out1 + downsampled_input
    -> out2

    out2 ->
    conv((3x3), (n,n))
    conv((3x3), (n,n))
    -> out3

    out3 + out2
    ->  output
    '''
    def __init__(
            self,
            inplanes: int,
            outplanes: int
    ) -> None:
        super().__init__()

        self.conv1 = ConvBlock(inplanes, outplanes, stride=2)
        self.conv2 = ConvBlock(outplanes, outplanes)

        self.conv3 = ConvBlock(outplanes, outplanes)
        self.conv4 = ConvBlock(outplanes, outplanes)

        self.downsample = ConvBlock(inplanes, outplanes, stride=2)
     
    def forward(
            self, 
            x: Tensor
    ) -> Tensor:
        out1 = self.conv1(x)
        out1 = self.conv2(out1)
        downsample = self.downsample(x)
        x2 = out1 + downsample

        out2 = self.conv3(x2)
        out2 = self.conv4(out2)
        return x2 + out2




class DecoderBlock(nn.Module):
    '''
    input ->
    conv((1,1), (m, m/4))
    full-conv((3,3), (m/4, m/4), *2)
    conv((1,1), (m/4, n))
    -> output
    '''
    def __init__(
            self,
            inplanes: int,
            outplanes: int
    ) -> None:
        super().__init__()

        m = inplanes
        m_4 = inplanes//4
        n = outplanes

        self.decode = nn.Sequential(
            ConvBlock(
                m,
                m_4,
                kernel_size=1,
                padding=0,
            ),
            FullConvBlock(
                m_4,
                m_4
            ),
            ConvBlock(
                m_4,
                n,
                kernel_size=1,
                padding=0
            )
        )
    
    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        x = self.decode(x)
        return x




class LinkNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_conv = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.encode1 = EncoderBlock(64, 64)
        self.encode2 = EncoderBlock(64, 128)
        self.encode3 = EncoderBlock(128, 256)
        self.encode4 = EncoderBlock(256, 512)

        self.decode4 = DecoderBlock(512, 256)
        self.decode3 = DecoderBlock(256, 128)
        self.decode2 = DecoderBlock(128, 64)
        self.decode1 = DecoderBlock(64, 64)

        self.output_conv_1 = nn.Sequential(
            FullConvBlock(64, 32),
            ConvBlock(32, 32)
        )
        self.output_conv_2 = FullConvBlock(
            32, 2, kernel_size=2, padding=0, output_padding=0
        )

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.input_conv(x)

        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)

        d4 = self.decode4(e4)
        d3 = self.decode3(d4+e3)
        d2 = self.decode2(d3+e2)
        d1 = self.decode1(d2+e1)

        x = self.output_conv_1(d1)
        x = self.output_conv_2(x, activate=False)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# test
if __name__ == '__main__':
    import numpy as np
    model = LinkNet()
    input = torch.Tensor(np.random.randint(0, 256, [1, 3, 256, 256]))
    output = model(input)
    print('\noutput: \n', output)
    print('\noutput size: \n', output.shape)


    expected_output_shape = [1, 2, 256, 256]

    if output.shape == torch.Size(expected_output_shape):
        print("\nSUCCESS!")
    else:
        print('\nThe size of the output tensor si expected to be {}, '\
              'but got {} instead'\
              ', please check the code.'.format([1, 3, 256, 256],list(output.shape)))