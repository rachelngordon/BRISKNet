import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from lsp import Project_inf, Wxs, Wtxs
from time import time
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.complex64

def to_torch_complex(x: torch.Tensor):
    """(B, 2, ...) real -> (B, ...) complex"""
    assert x.shape[1] == 2, (
        f"Input tensor must have 2 channels (real, imag), but got shape {x.shape}"
    )
    return torch.view_as_complex(rearrange(x, "b c ... -> b ... c").contiguous())


def from_torch_complex(x: torch.Tensor):
    """(B, ...) complex -> (B, 2, ...) real"""
    return rearrange(torch.view_as_real(x), "b ... c -> b c ...").contiguous()


# define LSFP-Net Block
class BasicBlock(nn.Module):
    @staticmethod
    def _to_3tuple(k):
        if isinstance(k, int):
            return (k, k, k)
        if isinstance(k, (list, tuple)) and len(k) == 3:
            return tuple(int(v) for v in k)
        raise ValueError("kernel_size must be an int or a tuple/list of three ints.")

    def __init__(self, lambdas, channels=32, kernel_size_L=3, kernel_size_S=3):
        super(BasicBlock, self).__init__()

        self.channels = channels
        self.kernel_size_L = self._to_3tuple(kernel_size_L)
        self.kernel_size_S = self._to_3tuple(kernel_size_S)
        self.padding_L = tuple(k // 2 for k in self.kernel_size_L)
        self.padding_S = tuple(k // 2 for k in self.kernel_size_S)

        self.lambda_L = nn.Parameter(torch.tensor([lambdas['lambda_L']]))
        self.lambda_S = nn.Parameter(torch.tensor([lambdas['lambda_S']]))
        self.lambda_spatial_L = nn.Parameter(torch.tensor([lambdas['lambda_spatial_L']]))
        self.lambda_spatial_S = nn.Parameter(torch.tensor([lambdas['lambda_spatial_S']]))

        self.gamma = nn.Parameter(torch.tensor([lambdas['gamma']]))
        self.lambda_step = nn.Parameter(torch.tensor([lambdas['lambda_step']]))


        # self.lambda_L = nn.Parameter(torch.tensor([0.0025]))
        # self.lambda_S = nn.Parameter(torch.tensor([0.05]))
        # self.lambda_spatial_L = nn.Parameter(torch.tensor([5e-2]))
        # self.lambda_spatial_S = nn.Parameter(torch.tensor([5e-2]))

        # self.gamma = nn.Parameter(torch.tensor([0.5]))
        # self.lambda_step = nn.Parameter(torch.tensor([1/10]))

        self.conv1_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 1, *self.kernel_size_L)))
        self.conv2_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_L)))
        self.conv3_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_L)))

        self.conv1_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_L)))
        self.conv2_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_L)))
        self.conv3_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(1, self.channels, *self.kernel_size_L)))

        self.conv1_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, 1, *self.kernel_size_S)))
        self.conv2_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_S)))
        self.conv3_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_S)))

        self.conv1_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_S)))
        self.conv2_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(self.channels, self.channels, *self.kernel_size_S)))
        self.conv3_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(1, self.channels, *self.kernel_size_S)))

    def forward(self, M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S, csmaps):

        c = self.lambda_step / self.gamma
        nx, ny, nt = M0.size()

        # gradient
        temp_data = torch.reshape(L + S, [nx, ny, nt])
        temp_data = param_E(inv=False, data=temp_data, smaps=csmaps).to(param_d.device)
        gradient = param_E(inv=True, data=temp_data - param_d, smaps=csmaps)
        gradient = torch.reshape(gradient, [nx * ny, nt]).to(param_d.device)

        # pb_L
        pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=self.padding_L)
        pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=self.padding_L)
        pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv3_backward_l, padding=self.padding_L)

        pb_L = torch.reshape(torch.squeeze(pb_L), [2, nx * ny, nt])
        pb_L = pb_L[0, :, :] + 1j * pb_L[1, :, :]

        # y_L
        y_L = L - self.gamma * gradient - self.gamma * pt_L - self.gamma * pb_L

        # pt_L

        # --- START OF THE ROBUST COMPLEX SVD FIX ---

        svd_input_complex = c * y_L + pt_L
        # print(f"Shape of svd_input_complex before processing: {svd_input_complex.shape}")

        # svd_input_complex = svd_input_complex.view(-1, n_frames, n_frames)
        
        # 1. Store the original magnitude and phase of the complex matrix.
        #    Add a small epsilon to the magnitude to prevent division by zero when calculating phase.
        svd_input_mag = svd_input_complex.abs() + 1e-8
        original_phase = svd_input_complex / svd_input_mag

        # print(f"Shape of svd_input_mag before regularization: {svd_input_mag.shape}")

        # epsilon = 1e-6
        # stable_input = svd_input_mag + epsilon * torch.eye(svd_input_mag.shape[-1], device=svd_input_mag.device)

        # print(f"Shape of stable_input after regularization: {stable_input.shape}")

        # noise_std = 1e-5 # A small standard deviation for the noise
        # noise = torch.randn_like(svd_input_mag) * noise_std

        noise_std = 1e-5 # A small standard deviation for the noise
        noise = torch.randn_like(svd_input_mag) * noise_std

        stable_svd_input = svd_input_mag + noise #epsilon

        

        # 2. Perform SVD on the REAL-VALUED magnitude matrix. 
        #    This completely avoids the complex svd_backward error.
        #    Using linalg.svd is fine here since the input is real.
        Ut, St, Vt = torch.linalg.svd(stable_svd_input, full_matrices=False)

        # 3. Apply the shrinkage/thresholding to the real singular values.
        #    (Project_inf operates on magnitudes, so this is correct).
        St_shrunk = Project_inf(St, self.lambda_L, to_complex=False)

        # 4. Reconstruct the new, thresholded MAGNITUDE matrix.
        pt_L_mag = Ut @ torch.diag_embed(St_shrunk) @ Vt

        # 5. Re-apply the original phase to our new magnitude matrix to get the
        #    final complex-valued update term.
        pt_L = pt_L_mag * original_phase


        # --- HOOK IMPLEMENTATION START ---

        # We define the hook function inside the forward pass so it has access to local variables if needed
        # def svd_grad_hook(grad):
        #     """
        #     This function will be called when the gradient for `pt_L` is computed.
        #     `grad` is the gradient tensor itself.
        #     """
        #     # Check if the gradient is valid
        #     if grad is not None:
        #         # We use .detach() to avoid modifying the computation graph during inspection
        #         grad_norm = grad.detach().norm(2).item()
        #         print(f"--- SVD GRADIENT HOOK ---")
        #         print(f"  Gradient norm of pt_L output: {grad_norm:.6e}")
        #         if grad_norm < 1e-7:
        #             print(f"  !!! WARNING: Gradient from SVD block is vanishing!")
        #         print(f"-------------------------")
        #     else:
        #         print("--- SVD GRADIENT HOOK: Gradient is None ---")

        # # Register the hook on the `pt_L` tensor. 
        # # It will be called automatically during the backward pass.
        # # We only want to do this during training.
        # if self.training:
        #     pt_L.register_hook(svd_grad_hook)

        # --- HOOK IMPLEMENTATION END ---


        # --- END OF THE ROBUST COMPLEX SVD FIX ---

        # svd_input = c * y_L + pt_L
        
        # Add a very small amount of random noise to the input matrix.
        # This is a standard technique to break ties in singular values and
        # prevent the gradient calculation from becoming ill-defined.
        # noise_std = 1e-3 # A small standard deviation for the noise
        # noise = torch.randn_like(svd_input) * noise_std

        
        # # Perform SVD on the slightly perturbed matrix
        # Ut, St, Vt = torch.svd(svd_input + noise)

        # Ut, St, Vt = torch.linalg.svd((c * y_L + pt_L), full_matrices=False)
        # temp_St = torch.diag(Project_inf(St, self.lambda_L))
        # pt_L = Ut.mm(temp_St).mm(Vt)

        # update p_L
        temp_y_L_input = torch.cat((torch.real(y_L), torch.imag(y_L)), 0).to(torch.float32)
        temp_y_L_input = torch.reshape(temp_y_L_input, [2, nx, ny, nt]).unsqueeze(1)
        temp_y_L = F.conv3d(temp_y_L_input, self.conv1_forward_l, padding=self.padding_L)
        temp_y_L = F.relu(temp_y_L)
        temp_y_L = F.conv3d(temp_y_L, self.conv2_forward_l, padding=self.padding_L)
        temp_y_L = F.relu(temp_y_L)
        temp_y_L_output = F.conv3d(temp_y_L, self.conv3_forward_l, padding=self.padding_L)

        temp_y_L = temp_y_L_output + p_L
        temp_y_L = temp_y_L[0, :, :, :, :] + 1j * temp_y_L[1, :, :, :, :]
        p_L = Project_inf(c * temp_y_L, self.lambda_spatial_L)

        # new pb_L
        p_L = torch.cat((torch.real(p_L), torch.imag(p_L)), 0).to(torch.float32)
        p_L = torch.reshape(p_L, [2, self.channels, nx, ny, nt])
        pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=self.padding_L)
        pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=self.padding_L)
        pb_L = F.relu(pb_L)
        pb_L_output = F.conv3d(pb_L, self.conv3_backward_l, padding=self.padding_L)

        pb_L = torch.reshape(pb_L_output, [2, nx * ny, nt])
        pb_L = pb_L[0, :, :] + 1j * pb_L[1, :, :]

        # L
        L = L - self.gamma * gradient - self.gamma * pt_L - self.gamma * pb_L

        # adjoint loss: adjloss_L = psi * x * y - psi_t * y * x
        # Reduce immediately to avoid keeping full per-layer adjloss tensors alive.
        adjloss_L = (temp_y_L_output * p_L - pb_L_output * temp_y_L_input).mean()

        # pb_S
        pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=self.padding_S)
        pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=self.padding_S)
        pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv3_backward_s, padding=self.padding_S)

        pb_S = torch.reshape(pb_S, [2, nx * ny, nt])
        pb_S = pb_S[0, :, :] + 1j * pb_S[1, :, :]

        # y_S
        y_S = S - self.gamma * gradient - self.gamma * Wtxs(pt_S) - self.gamma * pb_S

        # pt_S
        pt_S = Project_inf(c * Wxs(y_S) + pt_S, self.lambda_S)

        # update p_S
        temp_y_S_input = torch.cat((torch.real(y_S), torch.imag(y_S)), 0).to(torch.float32)
        temp_y_S_input = torch.reshape(temp_y_S_input, [2, nx, ny, nt]).unsqueeze(1)
        temp_y_S = F.conv3d(temp_y_S_input, self.conv1_forward_s, padding=self.padding_S)
        temp_y_S = F.relu(temp_y_S)
        temp_y_S = F.conv3d(temp_y_S, self.conv2_forward_s, padding=self.padding_S)
        temp_y_S = F.relu(temp_y_S)
        temp_y_S_output = F.conv3d(temp_y_S, self.conv3_forward_s, padding=self.padding_S)

        temp_y_Sp = temp_y_S_output + p_S
        temp_y_Sp = temp_y_Sp[0, :, :, :, :] + 1j * temp_y_Sp[1, :, :, :, :]
        p_S = Project_inf(c * temp_y_Sp, self.lambda_spatial_S)

        # new pb_S
        p_S = torch.cat((torch.real(p_S), torch.imag(p_S)), 0).to(torch.float32)
        p_S = torch.reshape(p_S, [2, self.channels, nx, ny, nt])
        pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=self.padding_S)
        pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=self.padding_S)
        pb_S = F.relu(pb_S)
        pb_S_output = F.conv3d(pb_S, self.conv3_backward_s, padding=self.padding_S)

        pb_S = torch.reshape(pb_S_output, [2, nx * ny, nt])
        pb_S = pb_S[0, :, :] + 1j * pb_S[1, :, :]

        # S
        S = S - self.gamma * gradient - self.gamma * Wtxs(pt_S) - self.gamma * pb_S

        # adjoint loss: adjloss_S = psi * x * y - psi_t * y * x
        # Reduce immediately to avoid keeping full per-layer adjloss tensors alive.
        adjloss_S = (temp_y_S_output * p_S - pb_S_output * temp_y_S_input).mean()

        return [L, S, adjloss_L, adjloss_S, pt_L, pt_S, p_L, p_S, self.lambda_L, self.lambda_S, self.lambda_spatial_L, self.lambda_spatial_S, self.gamma, self.lambda_step]


# define LSFP-Net
class LSFPNet(nn.Module):
    def __init__(self, LayerNo, lambdas, channels=32, kernel_size_L=3, kernel_size_S=3):
        super(LSFPNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.channels = channels

        for ii in range(LayerNo):
            onelayer.append(BasicBlock(lambdas, channels=self.channels, kernel_size_L=kernel_size_L, kernel_size_S=kernel_size_S))

        self.fcs = nn.ModuleList(onelayer)

    
    def plot_block_output(self, M0, L, S, iter, epoch, output_dir):

        time_frame_index = 3

        nx, ny, nt = M0.size()
        L = torch.reshape(L, [nx, ny, nt])
        S = torch.reshape(S, [nx, ny, nt])

        output_image = L + S

        # Create a 1x3 plot grid
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f"Basic Block Output at Time Frame {time_frame_index} and Iteration {iter}", fontsize=20)

        # --- Top Row: DL Reconstruction Comparison ---
        axes[0].imshow(np.abs(M0[..., time_frame_index].cpu().detach().numpy()), cmap='gray')
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(np.abs(L[..., time_frame_index].cpu().detach().numpy()), cmap='gray')
        axes[1].set_title("Background Component (L)")
        axes[1].axis("off")

        axes[2].imshow(np.abs(S[..., time_frame_index].cpu().detach().numpy()), cmap='gray')
        axes[2].set_title("Dynamic Component (S)")
        axes[2].axis("off")

        axes[3].imshow(np.abs(output_image[..., time_frame_index].cpu().detach().numpy()), cmap='gray')
        axes[3].set_title("Combined Image (L + S)")
        axes[3].axis("off")

        filename = os.path.join(output_dir, f'basic_block_output_{epoch}_iter{iter}.png')
        plt.savefig(filename)
        plt.close()


    def forward(self, M0, param_E, param_d, csmap, epoch, output_dir):

        # M0 = M0[..., 0] + 1j * M0[..., 1]
        # param_d = param_d[..., 0] + 1j * param_d[..., 1]

        nx, ny, nt = M0.size()
        L = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        S = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        pt_L = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        pt_S = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        p_L = torch.zeros([2, self.channels, nx, ny, nt], dtype=torch.float32).to(param_d.device)
        p_S = torch.zeros([2, self.channels, nx, ny, nt], dtype=torch.float32).to(param_d.device)

        layers_adj_L = []
        layers_adj_S = []

        for ii in range(self.LayerNo):
            [L, S, layer_adj_L, layer_adj_S, pt_L, pt_S, p_L, p_S, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step] = self.fcs[ii](M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S, csmap)
            layers_adj_L.append(layer_adj_L)
            layers_adj_S.append(layer_adj_S)

            if epoch is not None:
                self.plot_block_output(M0, L, S, iter=ii, epoch=epoch, output_dir=output_dir)
                

        L = torch.reshape(L, [nx, ny, nt])
        S = torch.reshape(S, [nx, ny, nt])

        return [L, S, layers_adj_L, layers_adj_S, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step]
    

# class EWrapper:
#     """
#     Make `physics` look like the callable interface expected by LSFP-Net:
#         E(inv=False, data=image)  -> k-space
#         E(inv=True , data=data )  -> image
#     Scaling is handled *outside*, so we just delegate.
#     """
#     def __init__(self, physics, csmap):
#         self.physics, self.csmap = physics, csmap

#     def __call__(self, *, inv: bool, data: torch.Tensor) -> torch.Tensor:
#         if inv:   # adjoint
#             data = from_torch_complex(data.unsqueeze(0))
#             data = self.physics.A_adjoint(data, self.csmap)
#             data = to_torch_complex(data).squeeze(0)
#             return rearrange(data, 't h w -> h w t')
        
#         else:     # forward
#             data = from_torch_complex(data.unsqueeze(0))
#             data = self.physics.A(data, self.csmap)
#             return to_torch_complex(data).squeeze(0)
        

class ArtifactRemovalLSFPNet(nn.Module):
    def __init__(self, backbone_net, output_dir, **kwargs):
        super(ArtifactRemovalLSFPNet, self).__init__()
        self.backbone_net = backbone_net
        self.output_dir = output_dir

    @staticmethod
    def _normalise_both(zf: torch.Tensor, data: torch.Tensor):
        """
        Per-dynamic-series max-magnitude scaling (paper default).
        Both `zf` (image) and `data` (k-space) share the SAME scalar.
        """
        scale = zf.abs().max()                       # scalar, grads OK
        return zf / scale, data / scale, scale
    
    @staticmethod
    def _normalise_indep(x: torch.Tensor):
        """
        Per-dynamic-series max-magnitude scaling (paper default).
        Both `zf` (image) and `data` (k-space) share the SAME scalar.
        """

        scale = torch.quantile(x.abs(), 0.99) + 1e-6
        if scale < 1e-6: # Handle case where input is all zeros
             scale = 1.0
        return x / scale, scale

    def forward(self, y, E, csmap, epoch=None, norm="both", **kwargs):

        # 1. Get the initial ZF recon. This defines our target energy/scale.
        x_init = E(inv=True, data=y, smaps=csmap)

        # 2. Permute and normalize the input for the network
        # print("--- Values Before Normalization --- ")
        # print("zf image min: ", torch.abs(x_init).min())
        # print("zf image max: ", torch.abs(x_init).max())
        # print("zf image mean: ", torch.abs(x_init).mean())
        # print("kspace min: ", torch.abs(y).min())
        # print("kspace max: ", torch.abs(y).max())
        # print("kspace mean: ", torch.abs(y).mean())
        if norm =="both":
            x_init_norm, y_norm, scale = self._normalise_both(x_init, y)
        elif norm == "independent":
            # 2. Normalize the image and k-space INDEPENDENTLY.
            x_init_norm, scale = self._normalise_indep(x_init)
            y_norm, scale_y = self._normalise_indep(y)
        elif norm == "none":
            x_init_norm = x_init
            y_norm = y
            scale = 1.0
        # print("--- Values After Normalization --- ")
        # print("zf image min: ", torch.abs(x_init_norm).min())
        # print("zf image max: ", torch.abs(x_init_norm).max())
        # print("zf image mean: ", torch.abs(x_init_norm).mean())
        # print("kspace min: ", torch.abs(y_norm).min())
        # print("kspace max: ", torch.abs(y_norm).max())
        # print("kspace mean: ", torch.abs(y_norm).mean())

        L, S, loss_layers_adj_L, loss_layers_adj_S, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step  = self.backbone_net(x_init_norm, E, y_norm, csmap, epoch, self.output_dir)


        loss_constraint_L = torch.square(torch.mean(loss_layers_adj_L[0])) / self.backbone_net.LayerNo
        loss_constraint_S = torch.square(torch.mean(loss_layers_adj_S[0])) / self.backbone_net.LayerNo

        for k in range(self.backbone_net.LayerNo - 1):
            loss_constraint_S += torch.square(torch.mean(loss_layers_adj_S[k + 1])) / self.backbone_net.LayerNo
            loss_constraint_L += torch.square(torch.mean(loss_layers_adj_L[k + 1])) / self.backbone_net.LayerNo


        recon = (L + S) * scale                 # rescale to original units

        # 4) stack & convert back to (B,2,T,H,W) float32
        x_hat = torch.stack((recon.real, recon.imag), dim=0).unsqueeze(0)  # (B,2,H,W,T)

        return x_hat, loss_constraint_L + loss_constraint_S, lambda_L, lambda_S, lambda_spatial_L, lambda_spatial_S, gamma, lambda_step
