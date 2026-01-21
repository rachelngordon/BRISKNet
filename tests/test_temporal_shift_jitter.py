import torch

from transform import TemporalShiftJitterAfterBaseline


def demo_temporal_shift_jitter_after_baseline() -> None:
    # Create a simple ramp so shifts are easy to verify.
    T = 8
    x = torch.arange(T, dtype=torch.float32).view(1, 1, T, 1, 1)

    transform = TemporalShiftJitterAfterBaseline(
        n_trans=1,
        max_shift=2,
        pre_contrast_baseline="first_frame",
        buffer_frames=1,
    )

    # Force a deterministic left shift by 1 frame on the enhancement phase.
    shifts = [torch.tensor(1)]
    x_aug = transform._transform(x, shifts=shifts)

    # Baseline and buffer are unchanged.
    assert torch.allclose(x_aug[:, :, :1], x[:, :, :1])
    assert torch.allclose(x_aug[:, :, 1:2], x[:, :, 1:2])

    # Enhancement is shifted with edge padding.
    enh = x[:, :, 2:]
    expected = torch.cat([enh[:, :, 1:], enh[:, :, -1:]], dim=2)
    assert torch.allclose(x_aug[:, :, 2:], expected)

    # Output shape is (n_trans, C, T, H, W) for B=1.
    assert x_aug.shape == (1, 1, T, 1, 1)


if __name__ == "__main__":
    demo_temporal_shift_jitter_after_baseline()
