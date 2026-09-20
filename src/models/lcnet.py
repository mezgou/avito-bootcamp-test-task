# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Архитектура PP-LCNet x1.0 для определения ориентации строк"""

# Адаптировано из PaddleClas, оставлена конфигурация textline_ori
# Убраны TheseusLayer, другие масштабы модели и загрузчик ImageNet-весов

import paddle
from paddle import Tensor, nn

BLOCKS = {
    "blocks2": ((3, 16, 32, 1, False),),
    "blocks3": (
        (3, 32, 64, (2, 1), False),
        (3, 64, 64, 1, False),
    ),
    "blocks4": (
        (3, 64, 128, (2, 1), False),
        (3, 128, 128, 1, False),
    ),
    "blocks5": (
        (3, 128, 256, (2, 1), False),
        (5, 256, 256, 1, False),
        (5, 256, 256, 1, False),
        (5, 256, 256, 1, False),
        (5, 256, 256, 1, False),
        (5, 256, 256, 1, False),
    ),
    "blocks6": (
        (5, 256, 512, (2, 1), True),
        (5, 512, 512, 1, True),
    ),
}


class ConvBNLayer(nn.Layer):
    """Свёртка, нормализация и активация Hardswish"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingNormal(),
            ),
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(
            out_channels,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(0.0),
            ),
            bias_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(0.0),
            ),
        )
        self.act = nn.Hardswish()

    def forward(self, x: Tensor) -> Tensor:
        """Преобразовать карту признаков"""

        return self.act(self.bn(self.conv(x)))


class SEModule(nn.Layer):
    """Взвесить каналы по глобальным признакам изображения"""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(channels, channels // 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(channels // 4, channels, kernel_size=1)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Применить веса каналов к входным признакам"""

        weights = self.avg_pool(x)
        weights = self.relu(self.conv1(weights))
        weights = self.hardsigmoid(self.conv2(weights))
        return x * weights


class DepthwiseSeparable(nn.Layer):
    """Поканальная свёртка и смешивание каналов"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int],
        use_se: bool,
    ) -> None:
        super().__init__()

        self.dw_conv = ConvBNLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.use_se = use_se
        if use_se:
            self.se = SEModule(in_channels)

        self.pw_conv = ConvBNLayer(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Извлечь пространственные и межканальные признаки"""

        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        return self.pw_conv(x)


class PPLCNet(nn.Layer):
    """PP-LCNet x1.0 с двумя выходами для классов 0 и 180 градусов"""

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = ConvBNLayer(3, 16, kernel_size=3, stride=2)

        for name, blocks in BLOCKS.items():
            stage = nn.Sequential(
                *[
                    DepthwiseSeparable(in_c, out_c, kernel, stride, use_se)
                    for kernel, in_c, out_c, stride, use_se in blocks
                ]
            )
            self.add_sublayer(name, stage)

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.last_conv = nn.Conv2D(512, 1280, kernel_size=1, bias_attr=False)
        self.act = nn.Hardswish()
        self.dropout = nn.Dropout(p=0.2, mode="downscale_in_infer")
        self.flatten = nn.Flatten(start_axis=1)
        self.fc = nn.Linear(1280, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Вернуть логиты классов без softmax"""

        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        x = self.avg_pool(x)
        x = self.act(self.last_conv(x))
        x = self.dropout(x)
        return self.fc(self.flatten(x))
