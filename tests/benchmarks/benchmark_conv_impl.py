from timeit import timeit

import torch

import continual as co
from continual.conv import _forward_step_impl


def benchmark_cpp_impl():
    RUNS = 10000
    C = 1
    T = 3
    S = 3
    L = 10
    H = 3
    B = 8

    with torch.no_grad():
        sample = torch.normal(mean=torch.zeros(B * L * C * H)).reshape((B, C, L, H))

        # Regular
        co_conv = co.Conv2d(
            in_channels=C, out_channels=1, kernel_size=(T, S), bias=True
        )

        if torch.cuda.is_available():
            print("Using CUDA")
            sample = sample.to(device="cuda")
            co_conv = co_conv.to(device="cuda")

        sample_step = sample[:, :, 0]

        # Regular
        co_conv.eval()
        co_conv.forward_steps(sample)  # warm up

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(f"Benchmarking Python _forward_step_py for {RUNS} runs")
        co_conv._forward_step_py(sample_step, co_conv.get_state())
        t_py = timeit(
            lambda: co_conv._forward_step_py(sample_step, co_conv.get_state()),
            number=RUNS,
        )
        print(f">> Time spent: {t_py}")
        if torch.cuda.is_available():
            print(f">> Max allocated memory: {torch.cuda.max_memory_allocated()}")
            torch.cuda.reset_peak_memory_stats()
        print("")

        print(f"Benchmarking C++ _forward_step_impl for {RUNS} runs")
        # Warm up
        _forward_step_impl(
            sample_step,
            co_conv.weight,
            co_conv.bias,
            co_conv.stride,
            co_conv.padding,
            co_conv._step_padding,
            co_conv.dilation,
            co_conv.groups,
            *(co_conv.get_state() or (None, None, None)),
        )
        t_cpp = timeit(
            lambda: _forward_step_impl(
                sample_step,
                co_conv.weight,
                co_conv.bias,
                co_conv.stride,
                co_conv.padding,
                co_conv._step_padding,
                co_conv.dilation,
                co_conv.groups,
                *(co_conv.get_state() or (None, None, None)),
            ),
            number=RUNS,
        )
        print(f">> Time spent: {t_cpp}")
        if torch.cuda.is_available():
            print(f">> Max allocated memory: {torch.cuda.max_memory_allocated()}")
            torch.cuda.reset_peak_memory_stats()
        print("")

        assert t_py > t_cpp
        savings_pct = (t_py - t_cpp) / t_py * 100
        print(f"Savings with C++ impl: {savings_pct:.2f}%\n")

        print(f"Benchmarking interface forward_step for {RUNS} runs")
        co_conv.forward_step(sample_step, co_conv.get_state())
        t_step = timeit(
            lambda: co_conv.forward_step(sample_step, co_conv.get_state()), number=RUNS
        )
        print(f">> Time spent: {t_step}")
        if torch.cuda.is_available():
            print(f">> Max allocated memory: {torch.cuda.max_memory_allocated()}")
            torch.cuda.reset_peak_memory_stats()
        print("")

        interface_pct = (t_step - t_cpp) / t_step * 100
        print(f"Relative cost of interface: {interface_pct:.2f}%")


if __name__ == "__main__":
    benchmark_cpp_impl()
