import json
import os
import shutil
from pathlib import Path

def load_prompts(file_path):
    """Load prompts from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, str):
            return json.loads(data)
        return data

def get_available_image_pairs(image_dir):
    """Get pairs of consecutive images from the directory."""
    image_dir = os.path.join(image_dir, 'images')
    if not os.path.exists(image_dir):
        print(f"Warning: Directory not found: {image_dir}")
        return {}
    
    files = sorted([f for f in os.listdir(image_dir) if f.startswith('000') and f.endswith('.png')])
    print(f"Found {len(files)} files in {image_dir}")
    
    pairs = {}
    for i in range(0, len(files) - 1, 2):
        base_num = int(files[i][3:5])  # Get number from 000xx.png
        next_file = f"{files[i][:3]}{base_num+1:02d}.png"
        if next_file in files:  # Removed base_num % 2 == 0 condition
            pairs[base_num] = (files[i], next_file)
    
    print(f"Found {len(pairs)} image pairs")
    return pairs

def find_prompt_in_list(prompt, prompt_list):
    """Find a prompt in the list, ignoring case and white space."""
    prompt = prompt.strip().lower()
    for i, p in enumerate(prompt_list):
        if p.strip().lower() == prompt:
            return i
    return -1

def map_and_replace_images(prompt_json_path, prompt2_json_path, 
                         d3po_image_dir, counting_image_dir, output_dir):
    """Main function to map and replace images."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load prompts
        print("Loading prompts...")
        prompts1 = load_prompts(prompt_json_path)
        prompts2 = load_prompts(prompt2_json_path)
        print(f"Loaded {len(prompts1)} prompts from first file")
        print(f"Loaded {len(prompts2)} prompts from second file")
        
        # Get available image pairs
        image_pairs = get_available_image_pairs(d3po_image_dir)
        
        # Get counting accuracy images
        files = sorted(os.listdir(counting_image_dir))
        counting_images = sorted([f for f in files if "counting_accuracy_" in f and f.endswith('.png')])
        print(f"Found {len(counting_images)} counting accuracy images")
        
        # Create mapping dictionary
        mapping = {}
        print("\nCreating mappings...")
        
        # For each counting accuracy image
        for counting_img in counting_images:
            # Extract the index from the filename
            idx = int(counting_img.replace("counting_accuracy_", "").replace(".png", ""))
            
            if idx < len(prompts2):
                prompt2 = prompts2[idx]
                # Find matching prompt in prompts1
                prompt1_idx = find_prompt_in_list(prompt2, prompts1)
                
                if prompt1_idx >= 0:
                    # Find corresponding image pair
                    for pair_num, (img1, img2) in image_pairs.items():
                        if pair_num == prompt1_idx * 2:  # Multiply by 2 since images are in pairs
                            mapping[counting_img] = (img1, img2)
                            print(f"Mapped {counting_img} to {img1}, {img2}")
                            print(f"  Prompt: {prompt2}")
                            break
        
        print(f"\nFound {len(mapping)} matches to process")
        
        # Copy and rename files
        for counting_img, (orig_img1, orig_img2) in mapping.items():
            src_img1 = os.path.join(d3po_image_dir, 'images', orig_img1)
            src_img2 = os.path.join(d3po_image_dir, 'images', orig_img2)
            src_counting = os.path.join(counting_image_dir, counting_img)
            
            dst_img1 = os.path.join(output_dir, orig_img1)
            dst_img2 = os.path.join(output_dir, orig_img2)
            
            print(f"\nProcessing {counting_img}:")
            print(f"Copying {src_img1} -> {dst_img1}")
            shutil.copy2(src_img1, dst_img1)
            
            print(f"Copying {src_img2} -> {dst_img2}")
            shutil.copy2(src_img2, dst_img2)
            
            print(f"Replacing {dst_img1} with {src_counting}")
            shutil.copy2(src_counting, dst_img1)
        
        return mapping
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def main():
    # Define paths
    prompt_json_path = "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/d3po/data/prompt.json"
    prompt2_json_path = "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data/counting_selected/annotations/counting_accuracy_prompts.json"
    d3po_image_dir = "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/d3po/data"
    counting_image_dir = "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data/counting_selected/images"
    output_dir = "mapped_images"
    
    print("Starting image mapping process...")
    mapping = map_and_replace_images(
        prompt_json_path,
        prompt2_json_path,
        d3po_image_dir,
        counting_image_dir,
        output_dir
    )
    
    print("\nFinal mapping created:")
    for counting_img, (orig_img1, orig_img2) in mapping.items():
        print(f"{counting_img} -> {orig_img1}, {orig_img2}")
    
    print("\nProcess completed!")

if __name__ == "__main__":
    main()