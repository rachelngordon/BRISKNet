import torch
import torch.nn as nn
from einops import rearrange
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SSIMVideoMetric(nn.Module):
    """
    A wrapper for the SSIM metric to handle 5D complex-valued video tensors
    and convert the similarity score into a minimization loss.

    This class performs the following steps:
    1. Converts the (B, 2, H, W, T) real/imag tensor to a (B, 1, T, H, W) magnitude tensor.
    2. Calculates the dynamic data_range (max - min) from the target tensor, which is
       crucial for accurate SSIM calculation.
    3. Reshapes the video into a batch of 2D images: ((B*T), 1, H, W).
    4. Computes the SSIM score using the calculated data_range.
    5. Returns (1.0 - SSIM) as the final loss value.
    """
    def __init__(self, gaussian_kernel=True, sigma=1.5, kernel_size=11):
        super().__init__()
        # Initialize the metric. The data_range will be set dynamically.
        self.ssim = StructuralSimilarityIndexMeasure(
            gaussian_kernel=gaussian_kernel,
            sigma=sigma,
            kernel_size=kernel_size
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure the metric is on the same device as the input tensors
        self.ssim.to(pred.device)

        # --- 1. Convert complex (real/imag) to magnitude ---
        # Input shape: (B, 2, H, W, T)
        pred_mag = torch.sqrt(pred[:, 0, ...]**2 + pred[:, 1, ...]**2).unsqueeze(1)
        target_mag = torch.sqrt(target[:, 0, ...]**2 + target[:, 1, ...]**2).unsqueeze(1)
        # Output shape: (B, 1, H, W, T)

        # --- 2. Calculate data_range dynamically from the target tensor ---
        # This is important for SSIM to work correctly across different images
        min_val = target_mag.min()
        max_val = target_mag.max()
        data_range = max_val - min_val

        # Handle the edge case of a black/zero-range image
        if data_range < 1e-6:
            # If the target is all zeros, perfect reconstruction should have SSIM=1, loss=0
            # We can compare with pred_mag directly here. If it's also all zeros, loss is 0.
            return torch.mean((pred_mag - target_mag)**2) # Fallback to MSE for zero-range images

        # Update the metric with the correct data_range for this specific batch
        self.ssim.data_range = data_range

        # --- 3. Reshape for SSIM: Fold Batch and Time dimensions ---
        # Input shape: (B, 1, H, W, T) -> rearrange -> (B, 1, T, H, W) -> flatten -> ((B*T), 1, H, W)
        pred_flat = rearrange(pred_mag, 'b c h w t -> (b t) c h w')
        target_flat = rearrange(target_mag, 'b c h w t -> (b t) c h w')

        # --- 4. Compute the SSIM score ---
        # The torchmetrics SSIM implementation returns a score in the range [-1, 1]
        ssim_score = self.ssim(pred_flat, target_flat)

        # --- 5. Convert similarity score to a minimization loss ---
        # We want to maximize SSIM, which is equivalent to minimizing (1 - SSIM)
        return 1.0 - ssim_score
    
class LPIPSVideoMetric(nn.Module):
    """
    A wrapper for the LPIPS metric to handle 5D complex-valued video tensors.
    
    This class performs the following steps:
    1. Converts the (B, 2, H, W, T) real/imag tensor to a (B, 1, T, H, W) magnitude tensor.
    2. **CORRECTED:** Finds the global min and max across BOTH prediction and target tensors
       to establish a shared data range for robust normalization.
    3. Normalizes both magnitude videos to the range [-1, 1].
    4. Clamps the results to [-1, 1] as a safeguard against floating point inaccuracies.
    5. Reshapes the video into a batch of 2D images: (B*T, 1, H, W).
    6. Repeats the single channel to create a 3-channel image for LPIPS.
    7. Computes the LPIPS score and returns the mean.
    """
    def __init__(self, net_type='alex'):
        super().__init__()
        # We perform normalization manually, so set normalize=False
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure input tensors are on the same device as the metric
        self.lpips.to(pred.device)

        # --- 1. Convert complex (real/imag) to magnitude ---
        pred_mag = torch.sqrt(pred[:, 0, ...]**2 + pred[:, 1, ...]**2).unsqueeze(1)
        target_mag = torch.sqrt(target[:, 0, ...]**2 + target[:, 1, ...]**2).unsqueeze(1)

        # --- 2. CORRECTED: Robust Normalization ---
        # Find the global min and max across both tensors to define a common range
        min_val = torch.min(pred_mag.min(), target_mag.min())
        max_val = torch.max(pred_mag.max(), target_mag.max())
        
        range_val = max_val - min_val
        # Avoid division by zero for black images
        if range_val < 1e-6:
            # If both images are essentially constant, the LPIPS is zero.
            return torch.tensor(0.0, device=pred.device)

        # Normalize both tensors using the common range
        pred_norm = 2 * (pred_mag - min_val) / range_val - 1
        target_norm = 2 * (target_mag - min_val) / range_val - 1

        # --- 3. Safeguard: Clamp values to ensure they are strictly in [-1, 1] ---
        pred_norm = torch.clamp(pred_norm, -1.0, 1.0)
        target_norm = torch.clamp(target_norm, -1.0, 1.0)

        # --- 4. Reshape for LPIPS: Fold Batch and Time dimensions ---
        pred_flat = rearrange(pred_norm, 'b c h w t -> (b t) c h w')
        target_flat = rearrange(target_norm, 'b c h w t -> (b t) c h w')

        # --- 5. Repeat single channel to get 3 channels ---
        pred_flat_rgb = pred_flat.repeat(1, 3, 1, 1)
        target_flat_rgb = target_flat.repeat(1, 3, 1, 1)

        # --- 6. Compute LPIPS ---
        return self.lpips(pred_flat_rgb, target_flat_rgb)


class HuberVideoMetric(nn.Module):
    """Huber loss over complex video tensors shaped (B, 2, H, W, T)."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=float(delta), reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Run in fp32 for numerical stability when training uses bf16/fp16.
        return self.huber(pred.float(), target.float())


class CharbonnierVideoMetric(nn.Module):
    """Charbonnier loss over complex video tensors shaped (B, 2, H, W, T)."""

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred.float() - target.float()
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()
