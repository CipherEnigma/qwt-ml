import argparse
import copy
import os.path
import random
import socket
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

import torch.distributed
import torch.utils.data
from timm.data.dataset import ImageDataset
from timm.utils import accuracy
from torch.utils.data import Dataset
from tqdm import tqdm

from quant import *
from utils import *
from utils.resnet import resnet101, resnet50, resnet18
from utils.utils import write, create_transform, create_loader, AverageMeter, broadcast_tensor_from_main_process, \
    gather_tensor_from_multi_processes, compute_quantized_params

HOST_NAME = socket.getfqdn(socket.gethostname())

torch.backends.cudnn.benchmark = True
LINEAR_COMPENSATION_SAMPLES = 512

model_path = {
    'resnet18': '/content/QwT/QwT-cls-RepQ-ViT/pretrained_weights/resnet18_imagenet.pth.tar',
    'resnet50': 'pretrained_weights/resnet50_imagenet.pth.tar',
    'resnet101': 'pretrained_weights/resnet101-63fe2227.pth'
}

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class CompensationBlock(nn.Module):#checks wether to turn on the compensation block or not 
    def __init__(self, W, b, r2_score, block, groups, linear_init=True, local_rank=0, block_id=None):
        super(CompensationBlock, self).__init__()
        self.block = block
        self.groups = groups

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1), W.size(2), W.size(3))))
        self.lora_bias = nn.Parameter(torch.zeros(b.size(0)))

        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            if local_rank == 0:
                _write('block {} using linear init'.format(block_id))
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
            if local_rank == 0:
                _write('block {} using lora init'.format(block_id))

    def forward(self, x):#x is the input feature map 
        out = self.block(x)#out is the output from the original quantized block

        B, C_X, H_X, W_X = x.size()
        _, C_Y, H_Y, W_Y = out.size()

        if (H_X == H_Y) and (W_X == W_Y):
            stride = 1
        elif (H_X // 2 == H_Y) and (W_X // 2 == W_Y):
            stride = 2
        else:
            raise NotImplementedError

        if self.training:
            qwt_out = F.conv2d(x, self.lora_weight, self.lora_bias, stride=stride, padding=int(self.lora_weight.size(-1) // 2), groups=self.groups)
        else:
            qwt_out = F.conv2d(x.half(), self.lora_weight.half(), None, stride=stride, padding=int(self.lora_weight.size(-1) // 2), groups=self.groups)
            qwt_out = qwt_out.float() + self.lora_bias.reshape(1, -1, 1, 1)

        out = out + qwt_out

        return out

def enable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, QuantConv2d) or isinstance(module, QuantLinear) or isinstance(module, QuantMatMul):
            module.set_quant_state(True, True)

def disable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, QuantConv2d) or isinstance(module, QuantLinear) or isinstance(module, QuantMatMul):
            module.set_quant_state(False, False)

class FeatureDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]
class RandomExpansionBlock(nn.Module):
    """
    ELM / RVFL Block: Uses fixed random projections to create non-linear features.
    Structure: x -> [x, ReLU(W_rand * x)] -> Linear -> Correction
    """
    def __init__(self, in_channels, out_channels, groups, expansion=8, stride=1, W_init=None, b_init=None, scale=1.0):
        super(RandomExpansionBlock, self).__init__()
        
        self.register_buffer('scale', torch.as_tensor(scale))
        
        # Hidden dimension for the random expansion
        hidden_dim = in_channels * expansion
        
        # 1. The Random Projection (Fixed, Untrained)
        # We register it as a buffer so it saves with the model but doesn't update
        self.register_buffer('rand_weight', torch.randn(hidden_dim, in_channels, 1, 1))
        self.register_buffer('rand_bias', torch.randn(hidden_dim))
        
        # Normalize random weights for stability
        self.rand_weight /= (in_channels ** 0.5)
        # Random bias helps shift the ReLUs to cover different parts of the "Fan"
        self.rand_bias /= 1.0 

        # 2. The Solver Layer (Linear Regression Result)
        # Input is: Original_Channels + Hidden_Channels
        solver_in_channels = in_channels + hidden_dim
        
        self.solver_conv = nn.Conv2d(solver_in_channels, out_channels, kernel_size=1, 
                                     stride=stride, groups=groups, bias=True)
        
        if W_init is not None:
            self.solver_conv.weight.data.copy_(W_init)
            self.solver_conv.bias.data.copy_(b_init)

    def forward(self, x):
        # Scale input for stability
        x_norm = x / self.scale
        
        # A. Random Feature Generation
        # Shape: [B, Hidden, H, W]
        # Note: We project pixel-wise (kernel=1) so stride doesn't matter here yet
        rand_out = F.conv2d(x_norm, self.rand_weight, self.rand_bias, stride=1, padding=0)
        rand_out = F.relu(rand_out) 
        
        # B. Feature Concatenation (Skip Connection)
        # We concatenate the raw input X with the new random features
        # Shape: [B, In + Hidden, H, W]
        features = torch.cat([x_norm, rand_out], dim=1)
        
        # C. Apply Solved Weights
        # This layer handles the stride (downsampling) 
        out = self.solver_conv(features)
        
        return out
    

def elm_regression(X, Y, expansion=8, kernel_size=1, groups=4, block_id=0, chunk_size=32):
    """
    Memory-Safe ELM Solver: Accumulates XTX and XTY in chunks to avoid OOM.
    """
    # Ensure inputs are on GPU
    X = gather_tensor_from_multi_processes(X, args.world_size).cuda()
    Y = gather_tensor_from_multi_processes(Y, args.world_size).cuda()
    
    B, C_X, H_X, W_X = X.size()
    _, C_Y, H_Y, W_Y = Y.size()
    
    # --- 1. Global Scaling ---
    # Must be calculated on the whole dataset once
    scale = X.abs().max() + 1e-6
    
    # --- 2. Generate Random Weights (Fixed for all chunks) ---
    hidden_dim = C_X * expansion
    rand_weight = torch.randn(hidden_dim, C_X, 1, 1, device=X.device)
    rand_bias = torch.randn(hidden_dim, device=X.device)
    
    # Normalize
    rand_weight /= (C_X ** 0.5)
    rand_bias /= 1.0
    
    # Determine Stride
    if (H_X == H_Y) and (W_X == W_Y):
        stride = 1
    elif (H_X // 2 == H_Y) and (W_X // 2 == W_Y):
        stride = 2
    else:
        stride = H_X // H_Y

    # Calculate Group Sizes
    # Note: The Input to the solver is (Original + Hidden)
    C_expanded_total = C_X + hidden_dim
    C_per_group_in = C_expanded_total // groups
    C_per_group_out = C_Y // groups
    
    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=int(kernel_size//2))

    weights_list = []
    bias_list = []

    # --- 3. Iterate Groups ---
    for g in range(groups):
        # Initialize Accumulators for this group
        # Size of X_bias is [L * C_per_group + 1]
        # We find the dimension by running a dummy pass or calculation
        # Calc feature dim: kernel_size*kernel_size * C_per_group_in + 1
        dim_w = (kernel_size * kernel_size * C_per_group_in) + 1
        
        XTX_sum = torch.zeros(dim_w, dim_w, device=X.device)
        XTY_sum = torch.zeros(dim_w, C_per_group_out, device=X.device)
        
        # --- 4. Chunked Accumulation (The Fix) ---
        for i in range(0, B, chunk_size):
            # Slice Mini-Batch
            X_chunk = X[i : i + chunk_size]
            Y_chunk = Y[i : i + chunk_size]
            current_B = X_chunk.size(0)
            
            # A. Expand Features (On the fly, for this chunk only)
            X_norm_chunk = X_chunk / scale
            rand_out = F.conv2d(X_norm_chunk, rand_weight, rand_bias, stride=1, padding=0)
            rand_out = F.relu(rand_out)
            X_expanded_chunk = torch.cat([X_norm_chunk, rand_out], dim=1)
            
            # B. Slice Group
            X_group = X_expanded_chunk[:, g * C_per_group_in : (g + 1) * C_per_group_in, :, :]
            Y_group = Y_chunk[:, g * C_per_group_out : (g + 1) * C_per_group_out, :, :]
            
            # C. Unfold & Flatten
            X_unfold = unfold(X_group) 
            # [Batch, Channels, Length] -> [Batch*Length, Channels]
            X_flat = X_unfold.permute(0, 2, 1).reshape(-1, X_unfold.shape[1])
            
            Y_flat = Y_group.view(current_B, C_per_group_out, -1)
            Y_flat = Y_flat.permute(0, 2, 1).reshape(-1, C_per_group_out)
            
            # D. Add Bias Term
            X_with_bias = torch.cat([X_flat, torch.ones(X_flat.shape[0], 1, device=X.device)], dim=1)
            
            # E. Accumulate into Global Sums
            # We assume the batch size fits in memory now
            XTX_sum += X_with_bias.T @ X_with_bias
            XTY_sum += X_with_bias.T @ Y_flat
            
            # Clean up immediate tensors
            del X_expanded_chunk, X_group, X_unfold, X_flat, X_with_bias
        
        # --- 5. Solve (Using Accumulated Sums) ---
        regularization = 1e-1
        XTX_reg = XTX_sum + regularization * torch.eye(XTX_sum.shape[0], device=X.device)
        
        W_solved = torch.inverse(XTX_reg) @ XTY_sum

        M_group = W_solved[:-1, :].T
        b_group = W_solved[-1, :]

        weights_list.append(M_group)
        bias_list.append(b_group)

    # Reconstruct Full Weights
    M_reshaped = torch.cat(weights_list, dim=0).view(C_Y, C_expanded_total // groups, kernel_size, kernel_size)
    b_final = torch.cat(bias_list, dim=0)

    # R2 Calc (Requires one more forward pass, but we can skip or chunk it if needed)
    # For OOM safety, let's do a quick chunked R2 or just return 0 if strict on memory.
    # Here is a simplified memory-safe R2 check (Just checks error variance)
    with torch.no_grad():
        # Re-run forward on chunks to get Y_pred
        ss_res = 0
        ss_tot = 0
        Y_mean = Y.mean(dim=0)
        
        for i in range(0, B, chunk_size):
            X_chunk = X[i : i + chunk_size]
            Y_chunk = Y[i : i + chunk_size]
            
            X_norm_chunk = X_chunk / scale
            rand_out = F.conv2d(X_norm_chunk, rand_weight, rand_bias, stride=1, padding=0)
            rand_out = F.relu(rand_out)
            X_expanded_chunk = torch.cat([X_norm_chunk, rand_out], dim=1)
            
            Y_pred_chunk = F.conv2d(X_expanded_chunk, M_reshaped, b_final, stride=stride, padding=kernel_size//2, groups=groups)
            
            ss_res += torch.sum((Y_chunk - Y_pred_chunk).pow(2))
            ss_tot += torch.sum((Y_chunk - Y_mean).pow(2))
            del X_expanded_chunk
            
        r2_score = 1 - ss_res / ss_tot

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f'ELM_Block : {block_id}      r2 : {r2_score:.3f}')

    return M_reshaped, b_final, r2_score, rand_weight, rand_bias, scale

class PolynomialCompensationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups, degree=2, stride=1, W_init=None, b_init=None, scale=1.0):
        super(PolynomialCompensationBlock, self).__init__()
        self.degree = degree
        
        # Store scale as a fixed tensor (buffer)
        self.register_buffer('scale', torch.as_tensor(scale))
        
        expanded_channels = in_channels * degree
        
        self.poly_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, 
                                   stride=stride, groups=groups, bias=True)
        
        if W_init is not None:
            self.poly_conv.weight.data.copy_(W_init)
            self.poly_conv.bias.data.copy_(b_init)

    def forward(self, x):
        features = []
        # Use the stored GLOBAL scale, not the batch max
        for d in range(1, self.degree + 1):
            features.append((x / self.scale).pow(d))
            
        x_poly = torch.cat(features, dim=1)
        out = self.poly_conv(x_poly)
        return out
@torch.no_grad()

def plot_quantization_error(y_full, y_quant, block_id, save_dir):
    """
    Plots visualization of the quantization error for a specific block.
    """
    # Convert to numpy and flatten for scatter/hist plots
    # We take a subset of points to keep plotting fast and readable
    y = y_full.detach().cpu().numpy().flatten()
    yz = y_quant.detach().cpu().numpy().flatten()
    
    # Sample 10,000 points randomly to avoid overcrowding the plot
    if len(y) > 10000:
        indices = np.random.choice(len(y), 10000, replace=False)
        y = y[indices]
        yz = yz[indices]
        
    error = y - yz
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Scatter Plot (Identity Line)
    # Ideal: All points lie exactly on the red diagonal line
    axs[0].scatter(y, yz, alpha=0.1, s=1, label='Quantized')
    axs[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal')
    axs[0].set_xlabel('Full Precision (y)')
    axs[0].set_ylabel('Quantized (yz)')
    axs[0].set_title(f'Block {block_id}: Input vs Output')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Error Histogram
    # Shows the distribution of the noise. Is it Gaussian? Is it skewed?
    axs[1].hist(error, bins=100, color='purple', alpha=0.7)
    axs[1].set_xlabel('Error (y - yz)')
    axs[1].set_ylabel('Count')
    axs[1].set_title(f'Block {block_id}: Error Distribution')
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Error vs Input Magnitude
    # This is crucial! It tells us if the error depends on the input value.
    # If you see a curve here, it means the error is Non-Linear (good for Polynomial/ELM).
    # If you see a flat blob, it's just random noise.
    axs[2].scatter(y, error, alpha=0.1, s=1, color='orange')
    axs[2].set_xlabel('Input Value (y)')
    axs[2].set_ylabel('Error (y - yz)')
    axs[2].set_title(f'Block {block_id}: Error Pattern')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'block_{block_id}_analysis.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved plot to {save_path}')
@torch.no_grad()

@torch.no_grad()
def generate_compensation_model_elm(q_model, train_loader, args):
    _write('Start to generate ELM-QwT model...')
    
    # --- Phase 1: Initial Data Collection ---
    torch.cuda.synchronize()
    output_t = torch.zeros(size=[0,], device=args.device)
    for i, (image, _) in tqdm(enumerate(train_loader)):
        image = image.cuda()
        t_out = q_model.forward_before_blocks(image)
        output_t = torch.cat([output_t, t_out.detach()], dim=0)
        torch.cuda.synchronize()
        if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
            break

    feature_set = FeatureDataset(output_t.detach().cpu())
    feature_loader = torch.utils.data.DataLoader(feature_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    output_previous = output_t
    sup_layers = [q_model.layer1, q_model.layer2, q_model.layer3, q_model.layer4]

    # --- Phase 2: Layer-wise Compensation ---
    for sup_id in range(len(sup_layers)):
        current_sup_layer = sup_layers[sup_id]
        for layer_id in range(len(current_sup_layer)):
            
            # Setup data
            feature_set.X = output_previous.detach().cpu()
            layer = current_sup_layer[layer_id]
            output_full_precision = torch.zeros(size=[0, ], device=args.device)
            output_quant = torch.zeros(size=[0, ], device=args.device)
            output_t_ = torch.zeros(size=[0, ], device=args.device)
            
            # Collect Block IO
            for i, t_out in tqdm(enumerate(feature_loader)):
                t_out = t_out.cuda()
                disable_quant(layer)
                full_precision_out = layer(t_out)
                enable_quant(layer)
                quant_out = layer(t_out)
                
                output_t_ = torch.cat([output_t_, t_out.detach()], dim=0)
                output_full_precision = torch.cat([output_full_precision, full_precision_out.detach()], dim=0)
                output_quant = torch.cat([output_quant, quant_out.detach()], dim=0)
                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size  // args.world_size - 1):
                    break
            
            # Calculate Target Error
            Y_diff = output_full_precision - output_quant
            
            groups = max(output_t_.size(1) // args.factor, 1)
            global_layer_id = sum(q_model.depths[:sup_id]) + layer_id
            out_channels_target = Y_diff.shape[1]

            # --- PLOTTING (Added Here) ---
            # This generates the plots for Loss and Error vs Input
            if args.local_rank == 0:
                plot_dir = os.path.join(args.log_dir, 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                plot_quantization_error(output_full_precision, output_quant, global_layer_id, plot_dir)
            # -----------------------------

            # --- ELM LOGIC ---
            EXPANSION = 8 
            
            # 1. Solve
            W, b, r2, rand_w, rand_b, scale = elm_regression(
                output_t_, Y_diff, 
                expansion=EXPANSION,
                kernel_size=args.kernel_size, 
                groups=groups, 
                block_id=global_layer_id
            )
            
            # 2. Detect Stride
            H_in = output_t_.shape[2]
            H_out = Y_diff.shape[2]
            stride = 2 if (H_in // 2 == H_out) else 1

            # 3. Insert Block
            new_block = RandomExpansionBlock(
                in_channels=output_t_.shape[1],
                out_channels=out_channels_target,
                groups=groups,
                expansion=EXPANSION,
                stride=stride,
                W_init=W, b_init=b,
                scale=scale
            )
            # Load random weights
            new_block.rand_weight.data.copy_(rand_w)
            new_block.rand_bias.data.copy_(rand_b)
            
            # 4. Wrap
            class HybridWrapper(nn.Module):
                def __init__(self, original_block, correction_block):
                    super().__init__()
                    self.block = original_block
                    self.correction = correction_block
                def forward(self, x):
                    return self.block(x) + self.correction(x)

            current_sup_layer[layer_id] = HybridWrapper(current_sup_layer[layer_id], new_block)
            del output_t_, output_full_precision, output_quant, Y_diff
            del W, b, rand_w, rand_b
            torch.cuda.empty_cache()
            # Update Outputs
            q_model.cuda()
            qwerty_layer = current_sup_layer[layer_id]
            output_previous = torch.zeros(size=[0, ], device=args.device)
            for i, t_out in tqdm(enumerate(feature_loader)):
                t_out = t_out.cuda()
                enable_quant(qwerty_layer)
                previous_out = qwerty_layer(t_out)
                output_previous = torch.cat([output_previous, previous_out.detach()], dim=0)
                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
                    break

    return q_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet18", choices=['resnet18', 'resnet50', 'resnet101'], help="model")
parser.add_argument('--data_dir', default='/opt/Dataset/ImageNet', type=str)

parser.add_argument('--w_bits', default=4, type=int, help='bit-precision of weights')
parser.add_argument('--a_bits', default=4, type=int, help='bit-precision of activations')
parser.add_argument('--start_block', default=0, type=int)
parser.add_argument('--kernel_size', default=1, type=int)
parser.add_argument('--factor', default=64, type=int)

parser.add_argument("--batch_size", default=32, type=int, help="batchsize of validation set")
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument("--seed", default=0, type=int, help="seed")

parser.add_argument("--local-rank", default=0, type=int)
args = parser.parse_args()

train_aug = 'large_scale_train'
test_aug = 'large_scale_test'
args.drop_path = 0.0
args.num_classes = 1000

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
crop_pct = 0.875

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
args.device = 'cuda:0'
args.world_size = 1
args.rank = 0  # global rank
if args.distributed:
    args.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()

assert args.rank >= 0


args.log_dir = os.path.join('checkpoint', args.model, 'QwTGroupConv', 'bs_{}_worldsize_{}_w_{}_a_{}_kernelsize_{}_factor_{}_startblock_{}_sed_{}' .format(args.batch_size, args.world_size, args.w_bits, args.a_bits, args.kernel_size, args.factor, args.start_block, args.seed))

args.log_file = os.path.join(args.log_dir, 'log.txt')


if args.local_rank == 0:
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if os.path.isfile(args.log_file):
        os.remove(args.log_file)
else:
    time.sleep(1)

torch.cuda.synchronize()

_write = partial(write, log_file=args.log_file)

if args.distributed:
    _write('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
else:
    _write('Training with a single process on 1 GPUs.')
assert args.rank >= 0


def main():
    if args.local_rank == 0:
        _write(args)

    seed(args.seed)

    if args.local_rank == 0:
        _write('dataset mean : {} & std : {}'.format(mean, std))

    dataset_train = ImageDataset(root=os.path.join(args.data_dir, 'train'), transform=create_transform(train_aug, mean, std, crop_pct))
    dataset_eval = ImageDataset(root=os.path.join(args.data_dir, 'val'), transform=create_transform(test_aug, mean, std, crop_pct))

    if args.local_rank == 0:
        _write('len of train_set : {}    train_transform : {}'.format(len(dataset_train), dataset_train.transform))
        _write('len of eval_set : {}    eval_transform : {}'.format(len(dataset_eval), dataset_eval.transform))


    loader_train = create_loader(
        dataset_train,
        batch_size=args.batch_size,
        is_training=True,
        re_prob=0.0,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        log_file=args.log_file,
        drop_last=True,
        local_rank=args.local_rank,
        persistent_workers=False
    )

    loader_eval = create_loader(
        dataset_eval,
        batch_size=args.batch_size,
        is_training=False,
        re_prob=0.,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        log_file=args.log_file,
        drop_last=False,
        local_rank=args.local_rank,
        persistent_workers=False
    )

    for data, _ in loader_train:
        calib_data = data.to(args.device)
        break

    broadcast_tensor_from_main_process(calib_data, args)
    _write('local_rank : {} calib_data shape : {} value : {}'.format(args.local_rank, calib_data.size(), calib_data[0, 0, 0, :5]))

    _write('Building model ...')
    if args.model == 'resnet18':
        model = resnet18(num_classes=args.num_classes, pretrained=False)
    elif args.model == 'resnet50':
        model = resnet50(num_classes=args.num_classes, pretrained=False)
    elif args.model == 'resnet101':
        model = resnet101(num_classes=args.num_classes, pretrained=False)
    else:
        raise NotImplementedError

    checkpoint = torch.load(model_path[args.model], map_location='cpu')
    model.load_state_dict(checkpoint)

    model.to(args.device)
    model.eval()

    fp32_model = copy.deepcopy(model)

    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model_resnet(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(args.device)
    q_model.eval()

    # Initial quantization
    _write('Performing initial quantization ...')
    set_quant_state(q_model, input_quant=True, weight_quant=True)

    with torch.no_grad():
        _ = q_model(calib_data)

    fp32_params = compute_quantized_params(fp32_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('FP32 model size is {:.3f}'.format(fp32_params))

    ptq_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('Percentile model size is {:.3f}'.format(ptq_params))

    top1_acc_eval = validate(fp32_model, loader_eval)
    _write('FP32 model   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

    top1_acc_eval = validate(q_model, loader_eval)
    _write('Percentile   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

    # --- CORRECTED FUNCTION CALL HERE ---
    q_model = generate_compensation_model_elm(q_model, loader_train, args)

    qwerty_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('QwT model size is {:.3f}'.format(qwerty_params))

    top1_acc_eval = validate(q_model, loader_eval)
    _write('Percentile + QwT   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

def validate(model, loader):
    top1_m = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):

            input = input.cuda()
            target = target.cuda()

            _, output = model(input)

            acc1, _ = accuracy(output, target, topk=(1, 5))

            top1_m.update(acc1.item(), output.size(0))

        top1_m.synchronize()

    _write('Test  Smples : {top1.count}    Acc@1: {top1.avg:>7.4f}'.format(top1=top1_m))
    return top1_m

if __name__ == '__main__':
    main()
