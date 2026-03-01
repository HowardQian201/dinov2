import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# 1. Setup Model
# ============================================================================
# RESOLUTION SETTINGS
# ============================================================================
# DINOv2 models use Vision Transformers with fixed patch size (14×14 pixels)
# 
# Resolution Options:
#   224×224: 16×16 = 256 patches (fast, standard)
#   518×518: 37×37 = 1369 patches (slower, much more detail)
#
# Note: Higher resolution = more patches = more computation but better detail
# The model uses interpolated positional embeddings for different resolutions
# ============================================================================

USE_HIGH_RES = False  # Set to True for 518×518, False for 224×224

# Load model
model_name = "facebook/dinov2-with-registers-giant"  # Largest model: ViT-g/14 with 1.1B parameters

# Set target resolution
if USE_HIGH_RES:
    target_size = 518
    print(f"🔧 Using model: {model_name} (ViT-g/14, 1.1B params) at {target_size}×{target_size} resolution")
    print(f"   Patches: 37×37 = 1369 patches | Patch size: 14×14 pixels")
else:
    target_size = 224
    print(f"🔧 Using model: {model_name} (ViT-g/14, 1.1B params) at {target_size}×{target_size} resolution")
    print(f"   Patches: 16×16 = 256 patches | Patch size: 14×14 pixels")

# Initialize processor with target size
processor = AutoImageProcessor.from_pretrained(
    model_name,
    size={"shortest_edge": target_size},
    crop_size={"height": target_size, "width": target_size}
)
model = AutoModel.from_pretrained(model_name, attn_implementation='eager')

# 2. Parse command line arguments and load images
parser = argparse.ArgumentParser(description="Compare DINOv2 patch embeddings between two images")
parser.add_argument("image1", nargs="?", default="1.png", help="Path to first image (default: 1.png)")
parser.add_argument("image2", nargs="?", default="2.png", help="Path to second image (default: 2.png)")
args = parser.parse_args()

image_path_1 = args.image1
image_path_2 = args.image2
image1 = Image.open(image_path_1).convert("RGB")
image2 = Image.open(image_path_2).convert("RGB")

# 3. Preprocess and Forward Pass for image 1
inputs1 = processor(images=image1, return_tensors="pt")
with torch.no_grad():
    outputs1 = model(**inputs1, output_attentions=False) # No need for attentions

# Preprocess and Forward Pass for image 2
inputs2 = processor(images=image2, return_tensors="pt")
with torch.no_grad():
    outputs2 = model(**inputs2, output_attentions=False) # No need for attentions

# Extract patch embeddings for image 1
# Token structure: [CLS] + [REG tokens] + [Patch tokens]
# For dinov2-with-registers, there are typically 4 register tokens
seq_len1 = outputs1.last_hidden_state.shape[1]
h_featmap1 = inputs1['pixel_values'].shape[-2] // 14
w_featmap1 = inputs1['pixel_values'].shape[-1] // 14
num_patches1 = h_featmap1 * w_featmap1
num_register_tokens1 = seq_len1 - num_patches1 - 1
patch_start_idx1 = 1 + num_register_tokens1
patch_embeddings1 = outputs1.last_hidden_state[0, patch_start_idx1:, :] # [num_patches, hidden_dim]

# Extract patch embeddings for image 2
seq_len2 = outputs2.last_hidden_state.shape[1]
h_featmap2 = inputs2['pixel_values'].shape[-2] // 14
w_featmap2 = inputs2['pixel_values'].shape[-1] // 14
num_patches2 = h_featmap2 * w_featmap2
num_register_tokens2 = seq_len2 - num_patches2 - 1
patch_start_idx2 = 1 + num_register_tokens2
patch_embeddings2 = outputs2.last_hidden_state[0, patch_start_idx2:, :] # [num_patches, hidden_dim]

# Get preprocessed images for visualization
preprocessed_img1 = inputs1['pixel_values'].squeeze().permute(1, 2, 0).cpu().numpy()
preprocessed_img2 = inputs2['pixel_values'].squeeze().permute(1, 2, 0).cpu().numpy()

# Normalize images for plotting (0-1 range)
preprocessed_img1 = (preprocessed_img1 - preprocessed_img1.min()) / (preprocessed_img1.max() - preprocessed_img1.min())
preprocessed_img2 = (preprocessed_img2 - preprocessed_img2.min()) / (preprocessed_img2.max() - preprocessed_img2.min())

# Get the actual preprocessed image sizes
img_height1, img_width1 = preprocessed_img1.shape[:2]
img_height2, img_width2 = preprocessed_img2.shape[:2]

patch_size1 = img_height1 // h_featmap1
patch_size2 = img_height2 // h_featmap2

# State variables for interaction
class State:
    last_patch_idx = None # Index of clicked patch in image 1

state = State()

def cosine_similarity(embedding1, embeddings2):
    """Calculates cosine similarity between a single embedding and a set of embeddings."""
    embedding1 = embedding1 / embedding1.norm(dim=-1, keepdim=True)
    embeddings2 = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)
    return (embedding1 @ embeddings2.T).cpu().numpy()

def update_visualization():
    """Update the visualization with current state."""
    if state.last_patch_idx is None:
        return
    
    # Get the embedding of the clicked patch from image 1
    clicked_embedding = patch_embeddings1[state.last_patch_idx]
    
    # Calculate cosine similarity with all patches in image 2
    similarity_map = cosine_similarity(clicked_embedding, patch_embeddings2)
    similarity_map = similarity_map.reshape(h_featmap2, w_featmap2)
    
    # Upsample for smooth overlay
    similarity_resized = np.array(Image.fromarray(similarity_map).resize((img_width2, img_height2), Image.BILINEAR))
    
    # Normalize for plotting
    similarity_resized = (similarity_resized - similarity_resized.min()) / (similarity_resized.max() - similarity_resized.min() + 1e-8)
    
    overlay2.set_data(similarity_resized)
    overlay2.set_clim(vmin=0, vmax=1)
    
    fig.canvas.draw_idle()

def onclick(event):
    """Handle mouse click events."""
    if event.inaxes == ax1 and event.xdata is not None and event.ydata is not None:
        # Convert click to patch coordinates for image 1
        px, py = int(event.xdata), int(event.ydata)
        
        px = max(0, min(px, img_width1 - 1))
        py = max(0, min(py, img_height1 - 1))
        
        pw = min(px // patch_size1, w_featmap1 - 1)
        ph = min(py // patch_size1, h_featmap1 - 1)
        idx = ph * w_featmap1 + pw
        
        state.last_patch_idx = idx
        
        marker1.set_data([pw * patch_size1 + patch_size1//2], 
                         [ph * patch_size1 + patch_size1//2])
        
        update_visualization()

# 4. Setup Interactive Visualization
# ============================================================================
print("\n📊 Setting up interactive visualization...")
print("   Click on any patch in the left image to see similarity heatmap on the right")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('DINOv2 Patch Embedding Similarity Visualization\nClick on left image to see similarities', 
             fontsize=14, fontweight='bold')

# Display image 1
ax1.imshow(preprocessed_img1)
ax1.set_title('Image 1 (Click on a patch)', fontsize=12)
ax1.axis('off')

# Display image 2
ax2.imshow(preprocessed_img2)
ax2.set_title('Image 2 (Similarity Heatmap)', fontsize=12)
ax2.axis('off')

# Initialize overlay for image 2 (hidden initially)
# Alpha controls transparency: lower = more see-through (0.0-1.0)
overlay2 = ax2.imshow(np.zeros((img_height2, img_width2)), 
                      cmap='jet', alpha=0.35, vmin=0, vmax=1)

# Add colorbar for similarity overlay
cbar = plt.colorbar(overlay2, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)

# Initialize marker for clicked patch on image 1
marker1, = ax1.plot([], [], 'r+', markersize=20, markeredgewidth=3)

# Connect click event
fig.canvas.mpl_connect('button_press_event', onclick)

plt.tight_layout()
print("✅ Visualization ready! Click on the left image to start.")
plt.show()