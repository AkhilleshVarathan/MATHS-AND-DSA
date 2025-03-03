import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Optional: D-Wave Imports (Only if using D-Wave)
try:
    from dwave.cloud import Client
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dimod import BinaryQuadraticModel

    # Enter your API token here (Replace with your actual token)
    API_TOKEN = "YOUR_API_TOKEN_HERE"
    client = Client.from_config(token=API_TOKEN)
    USE_DWAVE = True  # Set to False if no token
except ImportError:
    USE_DWAVE = False

# Paths
input_folder = r"D:\Archive (1)\scenes"
output_folder = r"D:\Quantum_Compression_Output"
os.makedirs(output_folder, exist_ok=True)

# Create separate folders for each step
folders = ["resized", "frqi", "qdct", "optimized", "reconstructed", "quality_metrics"]
for folder in folders:
    os.makedirs(os.path.join(output_folder, folder), exist_ok=True)

# Function to Load Images
def load_images(folder):
    images, filenames = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            filenames.append(filename)
        else:
            print(f"Warning: Could not load {filename}")
    return images, filenames

# Function to Resize Images
def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

# Function to Convert Image to FRQI
def image_to_frqi(image):
    norm_image = image / 255.0  # Normalize (0 to 1)
    angles = np.arccos(norm_image)  # Quantum angle representation
    return angles

# Function to Apply QDCT
def apply_qdct(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

# Function to Apply Inverse QDCT
def apply_iqdct(compressed_image):
    return idct(idct(compressed_image.T, norm='ortho').T, norm='ortho')

# Function to Save Image
def save_image(image, filename, step):
    save_path = os.path.join(output_folder, step, filename)
    plt.imsave(save_path, image, cmap='gray')

# Function to Use D-Wave for Optimization
def optimize_with_dwave(image):
    if not USE_DWAVE:
        print("Skipping D-Wave optimization (No API Token).")
        return image  # Return original image
    
    try:
        # Convert image to binary quadratic model (BQM)
        bqm = BinaryQuadraticModel.from_qubo(
            {(i, j): -image[i, j] for i in range(image.shape[0]) for j in range(image.shape[1])}
        )
        
        # Use D-Wave Sampler
        sampler = EmbeddingComposite(DWaveSampler(token=API_TOKEN))
        response = sampler.sample(bqm, num_reads=10)
        
        # Convert results back to an image
        best_sample = response.first.sample
        optimized_image = np.array([
            [best_sample.get(i * image.shape[1] + j, 0) for j in range(image.shape[1])]
            for i in range(image.shape[0])
        ])
        return optimized_image
    except Exception as e:
        print(f"D-Wave Optimization Error: {e}")
        return image  # Return original image in case of error

# Main Processing Loop
images, filenames = load_images(input_folder)

for img, filename in zip(images, filenames):
    print(f"Processing {filename}...")

    try:
        # Step 1: Resize
        resized_img = resize_image(img)
        save_image(resized_img, filename, "resized")

        # Step 2: Convert to FRQI
        frqi_img = image_to_frqi(resized_img)
        save_image(frqi_img, filename, "frqi")

        # Step 3: Apply QDCT
        qdct_img = apply_qdct(frqi_img)
        save_image(qdct_img, filename, "qdct")

        # Step 4: Use D-Wave for Optimization (Skip if No Token)
        optimized_img = optimize_with_dwave(qdct_img)
        save_image(optimized_img, filename, "optimized")

        # Step 5: Apply Inverse QDCT
        reconstructed_img = apply_iqdct(optimized_img)

        # Normalize reconstructed image properly
        reconstructed_img = reconstructed_img - np.min(reconstructed_img)
        reconstructed_img = reconstructed_img / np.max(reconstructed_img)

        save_image(reconstructed_img, filename, "reconstructed")

        # Step 6: Compute PSNR & SSIM
        original_norm = resized_img / 255.0
        reconstructed_norm = reconstructed_img  # Already normalized

        psnr_value = psnr(original_norm, reconstructed_norm)
        ssim_value = ssim(original_norm, reconstructed_norm, data_range=1.0)

        # Save Quality Metrics in Separate File
        quality_file = os.path.join(output_folder, "quality_metrics", f"{filename}_metrics.txt")
        with open(quality_file, "w") as f:
            f.write(f"PSNR = {psnr_value:.2f} dB\n")
            f.write(f"SSIM = {ssim_value:.4f}\n")

        print(f"Done: {filename} (PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f})")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Processing Complete! All outputs saved in:", output_folder)
