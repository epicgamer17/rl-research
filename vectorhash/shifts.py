import math
import torch
from ratslam_velocity_shift import inject_activity
from grid_module import GridModule
from vectorhash_functions import (
    generate_1d_gaussian_kernel,
    outer,
    expand_distribution,
    condense_distribution,
    calculate_shift_kernel,
)


class Shift:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, modules: list[GridModule], velocity: torch.Tensor):
        pass


class RollShift(Shift):
    def __init__(self, device=None):
        super().__init__(device)

    def __call__(self, modules, velocity):
        v_ = velocity.int()
        for module in modules:
            module.state = torch.roll(
                module.state,
                tuple([v_[i].item() for i in range(len(v_))]),
                dims=tuple(i for i in range(len(module.shape))),
            )


class RatShift(Shift):
    def __init__(self, device=None):
        super().__init__(device)

    def __call__(self, modules, velocity):
        assert len(velocity) == 3  # x, y, angular velocity

        for module in modules:
            speed = (velocity[0].item() ** 2 + velocity[1].item() ** 2) ** 0.5
            theta = math.atan2(velocity[1].item(), velocity[0].item())
            module.state = inject_activity(
                module.state, speed, theta, velocity[2].item()
            )


class ConvolutionalShift(Shift):
    def __init__(
        self, position_filter_std=0.1, angle_filter_std=6, filter_radius=6, device=None
    ):
        super().__init__(device)
        self.position_filter_std = position_filter_std
        self.angle_filter_std = angle_filter_std
        self.filter_radius = filter_radius

    def generate_kernels(self, velocity: torch.Tensor) -> list[torch.Tensor]:
        return [
            generate_1d_gaussian_kernel(
                self.filter_radius,
                mu=-v,
                sigma=self.position_filter_std if i < 2 else self.angle_filter_std,
                device=self.device,
            )
            for i, v in enumerate(velocity)
        ]

    def circular_conv(self, v: torch.Tensor, filter: torch.Tensor):
        """
        Input shape for v: (p)
        Input shape for filter: (f)

        """
        padded = torch.nn.functional.pad(
            v.unsqueeze(0),
            (len(filter) // 2, len(filter) // 2),
            mode="circular",
        )
        conv_result = torch.nn.functional.conv1d(
            padded.unsqueeze(0),
            filter.unsqueeze(0).unsqueeze(0),
            padding="valid",
        ).squeeze()

        conv_result = conv_result / conv_result.sum()  # prevent losing mass
        return conv_result

    def __call__(self, modules, velocity):
        filters = self.generate_kernels(velocity)
        dims = len(modules[0].shape)
        all_recovered_marginals = []  # (dim x module)

        for dim in range(dims):
            marginals = [module.get_marginal(dim) for module in modules]
            marginal_lengths = [len(marginal) for marginal in marginals]
            print(marginals)
            v = expand_distribution(marginals)
            v = self.circular_conv(v, filters[dim])
            recovered_marginals = condense_distribution(marginal_lengths, v)
            all_recovered_marginals.append(recovered_marginals)

        for i, module in enumerate(modules):
            marginals = [all_recovered_marginals[d][i] for d in range(dims)]
            module.state = module.state_from_marginals(marginals)


class ModularConvolutionalShift(Shift):
    def __init__(self, position_filter_std=1, angle_filter_std=12, device=None):
        self.position_filter_std = position_filter_std
        self.angle_filter_std = angle_filter_std
        super().__init__(device)

    def generate_kernel(self, module: GridModule, velocity: torch.Tensor):
        shape = tuple(module.shape)
        filters_1d = []
        for i in range(len(velocity)):
            filters_1d.append(
                calculate_shift_kernel(
                    radius=(shape[i] - 1) // 2,
                    shift=-(velocity[i] % module.shape[i]),
                    std=(
                        self.position_filter_std if i < 2 else self.angle_filter_std
                    ),
                    device=self.device,
                )
            )

        ret = outer(filters_1d)
        assert torch.isclose(
            ret.sum(), torch.ones(1, device=self.device)
        ), f"{ret.sum()}, {ret}"
        return ret

    def calculate_pad_tuple(self, module: GridModule):
        pad = []
        shape = tuple(module.shape)
        for l in reversed(shape):
            pad.append((l - 1) // 2)
            pad.append((l - 1) // 2)
        return tuple(pad)

    def __call__(self, modules, velocity):
        for module in modules:
            kernel = self.generate_kernel(module, velocity)
            pad = self.calculate_pad_tuple(module)
            state = module.state.unsqueeze(0)
            padded = torch.nn.functional.pad(state, pad, "circular")

            conv_result = (
                torch.nn.functional.conv3d(
                    padded.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)
                )
                .squeeze(0)
                .squeeze(0)
                .squeeze(0)
            )

            conv_result = conv_result / conv_result.sum()
            module.state = conv_result
