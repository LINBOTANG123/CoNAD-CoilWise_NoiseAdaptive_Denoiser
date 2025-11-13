
## Scanner-adaptive Coil-Level Denoising For Diffusion MRI Using Explicit Noise Priors

> **Abstract:** *Diffusion MRI (dMRI) is inherently limited by lowsignal-to-noise ratio (SNR), making denoising essential for preserving microstructural information. Most existing denoisers rely solely on the acquired noisy images, ignoring the significant variations in scanner- and coil-specific noise characteristics. In this work, we propose a scanner- and coil-adaptive denoising framework for multi-coil dMRI that leverages explicit noise priors and coil-level processing. Specifically, our model conditions the denoising network on the scanner’s noise standard deviation, estimated from a rapid “noise scan”. Furthermore, denoising is performed at the individual coil level prior to root-sum-of-squares combination, which dramatically reduces the noise floor while better preserving anatomical details. Experiments on in-vivo datasets demonstrate that our method achieves superior visual quality and quantitatively lower residual noise compared to state-of-the-art methods, without introducing over-smoothing or bias.* 
<hr />

## 🏗️ Installation

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run the model.

## ⏬ Download pretrained-weights

You can download the pretrained Restormer-based weights for the **CoNAD (Coil-wise Noise-Adaptive Denoiser)** model from the link below:

📥 [Download pretrained weights (Google Drive)](https://drive.google.com/drive/folders/1Sia6ka4ao_EVoMQ4OPFkWPNMPVJq4_vB?usp=sharing)

## 🧠 Run inference


### Case 1 — Inputs in *k-space*

If your MRI and noise data are in **k-space**, the script will automatically perform 2D inverse FFTs to convert them into the image domain before denoising:

```bash
python inference_denoising.py \
    --model_pth path_to_your_weights \
    --mri_file path_to_your_mri.mat \
    --mri_key mri_data_key \
    --noise_file path_to_your_noise_scan.mat \
    --noise_key noise_scan_key \
    --mri_is_kspace \
    --noise_is_kspace \
    --output_folder results
```

If your MRI and noise data are not in **k-space**:

```bash
python inference_denoising.py \
    --model_pth path_to_your_weights \
    --mri_file path_to_your_mri.mat \
    --mri_key mri_data_key \
    --noise_file path_to_your_noise_scan.mat \
    --noise_key noise_scan_key \
    --output_folder results
```

## 🧩 Train

You can train the our model using the [BasicSR](https://github.com/XPixelGroup/BasicSR) training framework.  
The training configuration file (`.yml`) defines all experiment settings including network architecture, dataset paths, optimizer, and scheduler parameters.

---

### Example: Train the model from a configuration file

```bash
python basicsr/train.py \
  -opt Denoising/Options/CoNAD.yml \
```

---

Acknowledgment:
Our denoising framework is built upon the excellent Restormer architecture and the BasicSR training library. We sincerely thank the authors for making their work and code publicly available.

