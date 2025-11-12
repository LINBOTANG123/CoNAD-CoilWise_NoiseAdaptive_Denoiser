import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

################################################################################
## PART 1: Main Restormer Inference (demo.py) ##################################
################################################################################

parser = argparse.ArgumentParser(description='Test Restormer on your own images')
parser.add_argument('--input_dir', default='./demo/degraded/', type=str, 
                    help='Directory of input images or path of single image')
parser.add_argument('--result_dir', default='./demo/restored/', type=str, 
                    help='Directory for restored results')
parser.add_argument('--task', required=True, type=str, help='Task to run', 
                    choices=['Motion_Deblurring',
                             'Single_Image_Defocus_Deblurring',
                             'Deraining',
                             'Real_Denoising',
                             'Gaussian_Gray_Denoising',
                             'Gaussian_Color_Denoising'])
parser.add_argument('--tile', type=int, default=None, 
                    help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, 
                    help='Overlapping of different tiles')
parser.add_argument('--noise_std', type=float, default=80.0, 
                    help='Noise standard deviation for Noise-Level-Aware testing')

args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

def get_weights_and_parameters(task, parameters):
    """Return model weights and updated parameters for each task."""
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == 'Gaussian_Color_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == 'Gaussian_Gray_Denoising':
        # Modify these paths/parameters as needed for your model
        weights = os.path.join('Denoising', 'pretrained_models', 'net_g_100000_frac.pth')
        parameters['inp_channels'] = 2  # image + noise map
        parameters['out_channels'] = 1
        parameters['LayerNorm_type'] = 'BiasFree'
    return weights, parameters

################################################################################
## PART 2: Metrics Calculation Function ########################################
################################################################################

def calculate_metrics(clean_folder, restored_folder, noised_folder, output_folder, noise_std=80):
    """Calculate PSNR/SSIM, save residual maps, and plot residual distributions."""
    os.makedirs(output_folder, exist_ok=True)

    clean_files = sorted(os.listdir(clean_folder))
    restored_files = sorted(os.listdir(restored_folder))
    noised_files = sorted(os.listdir(noised_folder))

    assert len(clean_files) == len(restored_files) == len(noised_files), "Mismatch in number of files"

    psnr_values = []
    ssim_values = []

    metrics_file = os.path.join(output_folder, "metrics.txt")

    with open(metrics_file, "w") as f:
        f.write("File-wise PSNR and SSIM Metrics:\n")
        f.write("================================\n")

        for clean_file, restored_file, noised_file in zip(clean_files, restored_files, noised_files):
            clean_path = os.path.join(clean_folder, clean_file)
            restored_path = os.path.join(restored_folder, restored_file)
            noised_path = os.path.join(noised_folder, noised_file)

            clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            restored_img = cv2.imread(restored_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            noised_img = cv2.imread(noised_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

            # Calculate PSNR and SSIM
            psnr_value = psnr(clean_img, restored_img, data_range=1.0)
            ssim_value = ssim(clean_img, restored_img, data_range=1.0)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

            # Save metrics to file
            f.write(f"{clean_file} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}\n")

            # Calculate residual map
            residual = restored_img - noised_img

            # Save residual map
            residual_map_path = os.path.join(output_folder, f"{os.path.splitext(clean_file)[0]}_residual.png")
            residual_img = ((residual - residual.min()) / (residual.max() - residual.min()) * 255).astype(np.uint8)
            cv2.imwrite(residual_map_path, residual_img)

            # Plot and save residual distribution
            residual_flat = residual.flatten()
            plt.figure(figsize=(10, 6))
            plt.hist(residual_flat, bins=100, alpha=0.6, label="Residual Distribution", density=True)

            # Add Gaussian noise distribution
            noise = np.random.normal(0, noise_std / 255.0, size=100000)
            plt.hist(noise, bins=100, alpha=0.6, label=f"Gaussian Noise (std={noise_std})", density=True, color='orange')

            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            plt.title(f"Residual vs Noise Distribution - {clean_file}")
            plt.grid(True)
            dist_plot_path = os.path.join(output_folder, f"{os.path.splitext(clean_file)[0]}_distribution.png")
            plt.savefig(dist_plot_path)
            plt.close()

            print(f"Processed {clean_file} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}")

        # Write average metrics to the file
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        f.write("\nAverage Metrics:\n")
        f.write("================\n")
        f.write(f"Average PSNR: {avg_psnr:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.3f}\n")

    print(f"Metrics saved to {metrics_file}. Residual maps and distribution plots saved to {output_folder}.")

################################################################################
## PART 3: Main Inference + Optional Metrics ###################################
################################################################################

def main():
    task    = args.task
    inp_dir = args.input_dir
    out_dir = os.path.join(args.result_dir, task)

    os.makedirs(out_dir, exist_ok=True)

    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

    # If a single file is provided as input_dir
    if any([inp_dir.endswith(ext) for ext in extensions]):
        files = [inp_dir]
    else:
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(inp_dir, '*.'+ext)))
        files = natsorted(files)

    if len(files) == 0:
        raise Exception(f'No files found at {inp_dir}')

    # Setup model
    parameters = {
        'inp_channels':3,
        'out_channels':3,
        'dim':48,
        'num_blocks':[4,6,6,8],
        'num_refinement_blocks':4,
        'heads':[1,2,4,8],
        'ffn_expansion_factor':2.66,
        'bias':False,
        'LayerNorm_type':'WithBias',
        'dual_pixel_task':False
    }

    weights, parameters = get_weights_and_parameters(task, parameters)
    load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    model.eval()

    img_multiple_of = 8

    print(f"\n ==> Running {task} with weights {weights}\n ")

    with torch.no_grad():
        for file_ in tqdm(files):
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            # Load image
            if task == 'Gaussian_Gray_Denoising':
                img = load_gray_img(file_)
            else:
                img = load_img(file_)

            input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

            # Pad the input if not multiple of 8
            height,width = input_.shape[2], input_.shape[3]
            H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
            padh = H-height if height % img_multiple_of != 0 else 0
            padw = W-width if width % img_multiple_of != 0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            # If tile is None, process entire image at once
            if args.tile is None:
                if task == 'Gaussian_Gray_Denoising':
                    # Noise-level aware approach (2 channels: grayscale + noise map)
                    noise_level = args.noise_std  # user-specified via argument
                    noise_level_map = torch.full_like(input_, fill_value=noise_level / 255.0)
                    input_with_noise = torch.cat((input_, noise_level_map), dim=1)
                    restored = model(input_with_noise)
                else:
                    # Normal approach (3-channel input)
                    restored = model(input_)
            else:
                # Tiled processing to handle large images
                b, c, h, w = input_.shape
                tile = min(args.tile, h, w)
                assert tile % 8 == 0, "tile size should be multiple of 8"
                tile_overlap = args.tile_overlap

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E = torch.zeros(b, parameters['out_channels'], h, w).type_as(input_)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch = model(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                        W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                restored = E.div_(W)

            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:,:,:height,:width]

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])

            f = os.path.splitext(os.path.split(file_)[-1])[0]
            save_path = os.path.join(out_dir, f+'.png')
            if task == 'Gaussian_Gray_Denoising':
                save_gray_img(save_path, restored)
            else:
                save_img(save_path, restored)

        print(f"\nRestored images are saved at {out_dir}\n")

    ################################################################################
    ## Optionally: Run Metrics Calculation after Restoration #######################
    ################################################################################
    # If you want to automatically compute metrics after inference, 
    # provide directories below (adjust as needed):
    if task == 'Gaussian_Gray_Denoising':
        # Example usage for integrated metrics:
        clean_folder   = "/home/lin/Research/denoise/Restormer/dataset/TEST_GRAY_CLEAN-20250120T051228Z-001/TEST_GRAY_CLEAN"
        restored_folder = out_dir  # The newly created folder with restored images
        noised_folder  = inp_dir
        metrics_out    = "results/metrics_80_frac_100k"

        # noise_std could be the same as args.noise_std (80 by default here)
        calculate_metrics(clean_folder, restored_folder, noised_folder, metrics_out, noise_std=args.noise_std)


if __name__ == "__main__":
    main()
