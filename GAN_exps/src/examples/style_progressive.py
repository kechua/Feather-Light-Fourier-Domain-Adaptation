import torch
from torch import Tensor, nn
from typing import List, Tuple, Dict
from gan.nn.stylegan.components import EqualLinear
from nn.progressiya.base import ProgressiveWithoutState, StateInjector, LastElementCollector, Progressive
from nn.progressiya.unet import ProgressiveSequential, InputFilterName, ZapomniKak


class InjectState(StateInjector):

    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]):
        return (torch.cat([state, args[0]], dim=-1), ), kw


class StyleTransform(nn.Module):

    def __init__(self):
        super().__init__()

        self.style_transform_1 = ProgressiveWithoutState[List[Tensor]](
            [EqualLinear(512, 512, activation='fused_lrelu')] \
            +[EqualLinear(512 * 2, 512, activation='fused_lrelu') for _ in range(13)],
            state_injector=InjectState()
        )

        self.style_transform_2 = ProgressiveWithoutState[List[Tensor]](
            [EqualLinear(512, 512, activation=None)] \
            +[EqualLinear(512 * 2, 512, activation=None) for _ in range(13)],
            state_injector=InjectState()
        )

        self.style_transform = ProgressiveSequential(
            self.style_transform_1,
            self.style_transform_2
        ).cuda()

    def forward(self, styles: Tensor):
        styles = [styles[:, i, ...] for i in range(styles.shape[1])]
        return torch.stack(tensors=self.style_transform(styles), dim=1)


class ConditionalStyleTransform(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = torch.nn.Embedding(2, 128).cuda()

        self.style_transform_1 = ProgressiveWithoutState[List[Tensor]](
            [EqualLinear(640, 512, activation='fused_lrelu')] \
            +[EqualLinear(640 + 512, 512, activation='fused_lrelu') for _ in range(13)],
            state_injector=InjectState()
        )

        self.style_transform_2 = ProgressiveWithoutState[List[Tensor]](
            [EqualLinear(512, 512, activation=None)] \
            +[EqualLinear(512 * 2, 512, activation=None) for _ in range(13)],
            state_injector=InjectState()
        )

        self.style_transform = ProgressiveSequential(
            self.style_transform_1,
            ZapomniKak('input'),
            self.style_transform_2
        ).cuda()

    def forward(self, styles: Tensor, cond: torch.Tensor):
        embed = self.embedding(cond)
        styles = [torch.cat((styles[:, i, ...], embed), dim=-1) for i in range(styles.shape[1])]
        return torch.stack(tensors=self.style_transform_2(self.style_transform_1(styles)), dim=1)
            #torch.stack(self.style_transform(input=styles), dim=1)


class Noise2Style(nn.Module):

    def __init__(self):
        super().__init__()

        self.style_list = [
            Progressive[List[Tensor]](
                [EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu')] \
                + [EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu') for _ in range(12)]
            )
        ]
        for i in range(4):
            self.style_list.append(
                ProgressiveWithoutState[List[Tensor]](
                    [EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu')] \
                    +[EqualLinear(512 * 2, 512, lr_mul=0.01, activation='fused_lrelu') for _ in range(13)],
                    state_injector=InjectState()
                )
            )

        self.style_transform = ProgressiveSequential(
            *self.style_list
        ).cuda()

    def forward(self, batch_size):
        input = torch.randn(batch_size, 512).cuda()
        for i in range(len(self.style_list)):
            input = self.style_list[i](input)
        return torch.stack(input, dim=1)
        #torch.stack(self.style_transform(torch.randn(batch_size, 512).cuda()), dim=1)


class StyleDisc(nn.Module):

    def __init__(self):
        super().__init__()

        self.progressija = ProgressiveWithoutState[List[Tensor]](
            [EqualLinear(512, 512, activation='fused_lrelu')] \
            +[EqualLinear(512 * 2, 512, activation='fused_lrelu') for _ in range(13)],
            state_injector=InjectState(),
            collector_class=LastElementCollector
        )

        self.head = nn.Sequential(EqualLinear(512, 512, activation='fused_lrelu'),
                             EqualLinear(512, 1, activation=None))

    def forward(self, styles: Tensor):
        styles = [styles[:, i, ...] for i in range(styles.shape[1])]
        return self.head(self.progressija(styles))