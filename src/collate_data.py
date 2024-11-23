import json
import os
import shutil
from pathlib import Path

def process_dataset(root_dir, output_dir):
    """
    Process the dataset directory structure and organize images.
    
    Args:
        root_dir (str): Path to the root directory containing prompt folders
        output_dir (str): Path to output directory for consolidated images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store prompt-to-image mapping
    prompt_image_map = {}
    
    # Walk through the directory structure
    for prompt_dir in Path(root_dir).glob('prompt_*'):
        prompt_id = prompt_dir.name.split('_')[1]
        metadata_file = prompt_dir / 'metadata.json'
        
        # Read metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            prompt = metadata['prompt']
            selected_images = metadata.get('selected_images', [])
            
            # Store in mapping
            prompt_image_map[prompt_id] = {
                'prompt': prompt,
                'images': []
            }
            
            # Process images in the prompt directory
            for img_file in prompt_dir.glob('selected_counting_*.png'):
                img_number = img_file.stem.split('_')[-1]
                
                # Copy and rename image
                new_filename = f'counting_accuracy_{prompt_id}.png'
                dest_path = os.path.join(output_dir, new_filename)
                
                # Copy the file if it doesn't already exist
                if not os.path.exists(dest_path):
                    shutil.copy2(img_file, dest_path)
                
                # Add image info to mapping
                prompt_image_map[prompt_id]['images'].append({
                    'original_name': img_file.name,
                    'new_name': new_filename
                })
    
    # Save the prompt-image mapping to a JSON file
    mapping_file = os.path.join(output_dir, 'prompt_image_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(prompt_image_map, f, indent=2)
    
    return prompt_image_map

def main():
    # Define paths (adjust these as needed)
    root_directory = '/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data/Selected_Images/Counting_Accuracy'  # Your root directory path
    output_directory = '/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data/consolidated_images'  # Where you want the images to be copied
    
    # Process the dataset
    prompt_map = process_dataset(root_directory, output_directory)
    
    # Print summary
    print(f"Processed {len(prompt_map)} prompts")
    print(f"Images consolidated in: {output_directory}")
    print(f"Mapping saved to: {os.path.join(output_directory, 'prompt_image_mapping.json')}")

if __name__ == "__main__":
    main()