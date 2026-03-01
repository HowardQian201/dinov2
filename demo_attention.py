import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import argparse
import sys
import os

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Interactive DINOv2 Attention Visualization',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python demo_attention.py image.jpg
  python demo_attention.py path/to/image.png --high-res
  python demo_attention.py image.jpg --model facebook/dinov2-base
    """
)
parser.add_argument('image', type=str, nargs='?', default='1.png',
                    help='Path to input image (default: 1.png)')
parser.add_argument('--high-res', action='store_true',
                    help='Use 518×518 resolution instead of 224×224 (slower but more detailed)')
parser.add_argument('--model', type=str, default='facebook/dinov2-with-registers-giant',
                    help='Model name (default: facebook/dinov2-with-registers-giant)')

args = parser.parse_args()

# Validate image path
if not os.path.exists(args.image):
    print(f"❌ Error: Image file '{args.image}' not found!")
    sys.exit(1)

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

USE_HIGH_RES = args.high_res

# Load model
model_name = args.model

print(f"\n{'='*70}")
print(f"📸 Input image: {args.image}")
print(f"🤖 Model: {model_name}")

# Set target resolution
if USE_HIGH_RES:
    target_size = 518
    print(f"📐 Resolution: {target_size}×{target_size} (HIGH-RES)")
    print(f"   Patches: 37×37 = 1369 patches | Patch size: 14×14 pixels")
else:
    target_size = 224
    print(f"📐 Resolution: {target_size}×{target_size} (STANDARD)")
    print(f"   Patches: 16×16 = 256 patches | Patch size: 14×14 pixels")

print(f"{'='*70}")

# Initialize processor with target size
processor = AutoImageProcessor.from_pretrained(
    model_name,
    size={"shortest_edge": target_size},
    crop_size={"height": target_size, "width": target_size}
)
model = AutoModel.from_pretrained(model_name, attn_implementation='eager')

# 2. Load image from local file
image_path = args.image
print(f"\n🔄 Loading image: {image_path}")
image = Image.open(image_path).convert("RGB")
print(f"✅ Image loaded: {image.size[0]}×{image.size[1]} pixels")

# 3. Preprocess and Forward Pass
# Processor will resize to the configured size
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# Print model output information
print("\n" + "="*60)
print("DINO MODEL OUTPUT SHAPES")
print("="*60)
print(f"Input shape: {inputs['pixel_values'].shape}")
print(f"  (batch, channels, height, width)")

print(f"\nFeature map (last_hidden_state): {outputs.last_hidden_state.shape}")
print(f"  (batch, sequence_length, hidden_dim)")

print(f"\nNumber of transformer layers: {len(outputs.attentions)}")
print(f"Attention shape (last layer): {outputs.attentions[-1].shape}")
print(f"  (batch, num_heads, sequence_length, sequence_length)")
print("="*60)

# 4. Extract Attention
# Shape: (batch, num_heads, sequence_length, sequence_length)
# We take the LAST layer's attention
attentions = outputs.attentions[-1] 

# 5. Process the [CLS] token attention
# Token structure: [CLS] + [REG tokens] + [Patch tokens]
# For dinov2-with-registers, there are typically 4 register tokens
nh = attentions.shape[1] # Number of heads (12 for Base)
seq_len = attentions.shape[-1]  # Total sequence length

# Calculate number of patches
h_featmap = inputs['pixel_values'].shape[-2] // 14  # height
w_featmap = inputs['pixel_values'].shape[-1] // 14  # width
num_patches = h_featmap * w_featmap

# Calculate number of register tokens
num_register_tokens = seq_len - num_patches - 1  # subtract patches and [CLS]
print(f"\nTOKEN BREAKDOWN:")
print(f"Total tokens: {seq_len}")
print(f"  [CLS] token: 1")
print(f"  Register tokens: {num_register_tokens}")
print(f"  Patch tokens: {num_patches} ({h_featmap}x{w_featmap})")

# Show feature map breakdown
hidden_dim = outputs.last_hidden_state.shape[-1]
print(f"\nFEATURE MAP BREAKDOWN:")
print(f"  CLS features: outputs.last_hidden_state[0, 0, :] -> shape ({hidden_dim},)")
print(f"  Register features: outputs.last_hidden_state[0, 1:{1+num_register_tokens}, :] -> shape ({num_register_tokens}, {hidden_dim})")
print(f"  Patch features: outputs.last_hidden_state[0, {1+num_register_tokens}:, :] -> shape ({num_patches}, {hidden_dim})")
print(f"  Spatial patch grid: Can be reshaped to ({h_featmap}, {w_featmap}, {hidden_dim})")
print("="*60)

# Extract patch-to-patch attention (skip [CLS] and register tokens)
# Tokens: [0: CLS] [1:1+num_register_tokens: registers] [1+num_register_tokens:end: patches]
patch_start_idx = 1 + num_register_tokens

# Extract attention between patches only: [num_patches, num_patches] for each head
# Shape: [batch, heads, seq_len, seq_len]
# We want: [heads, num_patches, num_patches]
patch_to_patch_attn = attentions[0, :, patch_start_idx:, patch_start_idx:]  # [heads, patches, patches]

# Extract CLS attention to patches: [heads, num_patches]
cls_to_patch_attn = attentions[0, :, 0, patch_start_idx:]  # [heads, patches]

print(f"\nAttention shapes:")
print(f"  Patch-to-patch: {patch_to_patch_attn.shape}")
print(f"  CLS-to-patch: {cls_to_patch_attn.shape}")
print(f"  Number of attention heads: {nh}")

# 6. Interactive Visualization with Alpha Blending
preprocessed_img = inputs['pixel_values'].squeeze().permute(1, 2, 0).cpu().numpy()
# Normalize image for plotting (0-1 range)
preprocessed_img = (preprocessed_img - preprocessed_img.min()) / (preprocessed_img.max() - preprocessed_img.min())

# Get the actual preprocessed image size
img_height, img_width = preprocessed_img.shape[:2]
print(f"\nVisualization image size: {img_width}×{img_height}")

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(preprocessed_img)

# Initialize overlay: Use a dummy array then update on click
# Set vmin=0, vmax=1 for proper color scaling
overlay = ax.imshow(np.zeros((img_height, img_width)), cmap='jet', alpha=0.5, vmin=0, vmax=1, 
                     extent=(0, img_width, img_height, 0))
marker, = ax.plot([], [], 'r+', markersize=20, markeredgewidth=3)
ax.axis('off')

patch_size = img_height // h_featmap  # Patch size in pixels (14 for 224x224, 14 for 518x518)

# State variables for interaction
class State:
    current_head = -1  # -1 means average, 0-11 for individual heads
    last_patch_idx = None
    last_ph = None
    last_pw = None
    cls_mode = False  # False = patch-to-patch, True = CLS-to-patch

state = State()

def get_attention_for_head(head_idx, cls_mode=False):
    """Get attention for a specific head or average.
    
    Args:
        head_idx: -1 for average, 0-11 for specific head
        cls_mode: If True, returns CLS attention [num_patches]. 
                  If False, returns patch-to-patch attention [num_patches, num_patches]
    """
    if cls_mode:
        # CLS to patch attention: [num_patches]
        if head_idx == -1:
            return cls_to_patch_attn.mean(dim=0).cpu().numpy()
        else:
            return cls_to_patch_attn[head_idx].cpu().numpy()
    else:
        # Patch to patch attention: [num_patches, num_patches]
        if head_idx == -1:
            return patch_to_patch_attn.mean(dim=0).cpu().numpy()
        else:
            return patch_to_patch_attn[head_idx].cpu().numpy()

def update_visualization():
    """Update the visualization with current state."""
    if state.cls_mode:
        # CLS mode: show CLS attention to all patches
        attn_vector = get_attention_for_head(state.current_head, cls_mode=True)
        attn_map = attn_vector.reshape(h_featmap, w_featmap)
        
        print(f"CLS attention stats - min: {attn_map.min():.6f}, max: {attn_map.max():.6f}, mean: {attn_map.mean():.6f}")
        
        # Hide marker in CLS mode
        marker.set_data([], [])
    else:
        # Patch mode: show selected patch's attention to other patches
        if state.last_patch_idx is None:
            # No patch selected yet - show a message and return
            ax.set_title("Click a patch to see its attention\n(← → switch heads | C toggle CLS mode)", 
                         fontsize=14, fontweight='bold')
            return
        
        attn_matrix = get_attention_for_head(state.current_head, cls_mode=False)
        attn_map = attn_matrix[state.last_patch_idx].reshape(h_featmap, w_featmap)
        
        print(f"Patch attention stats - min: {attn_map.min():.6f}, max: {attn_map.max():.6f}, mean: {attn_map.mean():.6f}")
        
        # Show marker in patch mode
        if state.last_ph is not None and state.last_pw is not None:
            marker.set_data([state.last_pw * patch_size + patch_size//2], 
                          [state.last_ph * patch_size + patch_size//2])
    
    # Upsample for smooth overlay
    attn_resized = np.array(Image.fromarray(attn_map).resize((img_width, img_height), Image.BILINEAR))
    
    # Dynamic Normalization
    attn_resized = attn_resized / (attn_resized.max() + 1e-8)
    
    print(f"After normalization - min: {attn_resized.min():.6f}, max: {attn_resized.max():.6f}")
    
    # Update plot elements
    overlay.set_data(attn_resized)
    overlay.set_clim(vmin=0, vmax=1)
    
    # Update title
    head_str = "Average (all heads)" if state.current_head == -1 else f"Head {state.current_head + 1}/{nh}"
    if state.cls_mode:
        mode_str = "[CLS] token attention to patches"
    else:
        mode_str = f"Patch [{state.last_ph}, {state.last_pw}] attention" if state.last_ph is not None else "Patch attention"
    
    ax.set_title(f"{mode_str} | {head_str}\n(← → switch heads | C toggle CLS mode)", 
                 fontsize=14, fontweight='bold')
    fig.canvas.draw_idle()

def onclick(event):
    """Handle mouse click events."""
    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        # Clicking switches to patch mode if in CLS mode
        if state.cls_mode:
            state.cls_mode = False
            print(f"\n🔄 Switched to Patch mode")
        
        # Convert click to patch coordinates
        px, py = int(event.xdata), int(event.ydata)
        
        # Clamp to valid range
        px = max(0, min(px, img_width - 1))
        py = max(0, min(py, img_height - 1))
        
        pw = min(px // patch_size, w_featmap - 1)
        ph = min(py // patch_size, h_featmap - 1)
        idx = ph * w_featmap + pw
        
        print(f"🖱️ Clicked pixel ({px}, {py}) -> Patch [{ph}, {pw}] (index: {idx})")
        
        # Update state
        state.last_patch_idx = idx
        state.last_ph = ph
        state.last_pw = pw
        
        # Update marker position
        marker.set_data([pw * patch_size + patch_size//2], [ph * patch_size + patch_size//2])
        
        # Update visualization
        update_visualization()

def onkey(event):
    """Handle keyboard events for switching heads and modes."""
    if event.key == 'c' or event.key == 'C':
        # Toggle CLS mode
        state.cls_mode = not state.cls_mode
        mode_name = "CLS mode" if state.cls_mode else "Patch mode"
        print(f"\n🔄 Switched to {mode_name}")
        update_visualization()
    elif event.key == 'right':
        # Next head (cycle: average -> head 0 -> head 1 -> ... -> head 11 -> average)
        if state.current_head == -1:
            state.current_head = 0  # From average to first head
        elif state.current_head == nh - 1:
            state.current_head = -1  # From last head to average
        else:
            state.current_head += 1
        update_visualization()
    elif event.key == 'left':
        # Previous head
        if state.current_head == -1:
            state.current_head = nh - 1  # From average to last head
        elif state.current_head == 0:
            state.current_head = -1  # From first head to average
        else:
            state.current_head -= 1
        update_visualization()
    elif event.key == 'a':
        # Jump to average
        state.current_head = -1
        update_visualization()

# Connect event handlers
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)

# Update initial title
ax.set_title("Click a patch to see its 'view' of the image\n(Use ← → keys to switch heads | Press C to toggle CLS mode)", 
             fontsize=14, fontweight='bold')

print("\n✓ Interactive Visualization Ready!")
print("━" * 60)
print("🖱️  Click on image to select a patch (Patch mode)")
print("⌨️  Press 'C' to toggle between Patch mode / CLS mode")
print("⌨️  Use ← → arrow keys to switch between attention heads")
print("⌨️  Press 'A' to jump to average across all heads")
print("")
print("📊 Modes:")
print("   • Patch mode: Shows how a selected patch attends to others")
print("   • CLS mode: Shows how the [CLS] token attends to all patches")
print("━" * 60)
plt.show()