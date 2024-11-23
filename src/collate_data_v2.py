import json
import os
import shutil
from pathlib import Path

def reorganize_dataset(root_dir, output_dir, json_output_dir):
    """
    Reorganize the dataset with a new structure where prompts are stored in a list
    and images are named based on the prompt's index in that list.
    
    Args:
        root_dir (str): Path to the root directory containing prompt folders
        output_dir (str): Path to output directory for consolidated images
        json_output_dir (str): Path to output directory for JSON file
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Initialize the new structure
    dataset_structure = {
        "counting_accuracy": []  # List to store all prompts
    }
    
    # Temporary dictionary to store prompt_id -> prompt mapping
    prompt_map = {}
    
    # First pass: collect all prompts
    for prompt_dir in Path(root_dir).glob('prompt_*'):
        prompt_id = prompt_dir.name.split('_')[1]
        metadata_file = prompt_dir / 'metadata.json'
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                prompt_map[int(prompt_id)] = metadata['prompt']
    
    # Sort prompt_ids numerically to ensure consistent ordering
    sorted_prompt_ids = sorted(prompt_map.keys())
    
    # Create the ordered list of prompts
    for prompt_id in sorted_prompt_ids:
        dataset_structure["counting_accuracy"].append(prompt_map[prompt_id])
    
    # Second pass: copy and rename images based on prompt index
    for prompt_dir in Path(root_dir).glob('prompt_*'):
        prompt_id = int(prompt_dir.name.split('_')[1])
        
        # Find the index of this prompt in our list
        prompt_index = sorted_prompt_ids.index(prompt_id)
        
        # Process images in the prompt directory
        for img_file in prompt_dir.glob('selected_counting_*.png'):
            # Create new filename based on prompt index
            new_filename = f'counting_accuracy_{prompt_index}.png'
            dest_path = os.path.join(output_dir, new_filename)
            
            # Copy the file
            shutil.copy2(img_file, dest_path)
            break  # Only copy the first image for each prompt
    
    # Save the new structure to a JSON file
    json_file_path = os.path.join(json_output_dir, 'counting_accuracy_prompts.json')
    with open(json_file_path, 'w') as f:
        json.dump(dataset_structure, f, indent=2)
    
    return len(dataset_structure["counting_accuracy"])

def main():
    # Define paths (adjust these as needed)
    base_path = "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data"
    root_directory = f'{base_path}/Selected_Images/Counting_Accuracy'  # Your root directory path
    image_output_directory = f'{base_path}/counting/images'  # Where you want the images to be copied
    json_output_directory = f'{base_path}/counting/annotations'  # New path for JSON
    
    # Process the dataset
    num_prompts = reorganize_dataset(root_directory, 
                                   image_output_directory, 
                                   json_output_directory)
    
    # Print summary
    print(f"Processed {num_prompts} prompts")
    print(f"Images saved in: {image_output_directory}")
    print(f"JSON file saved in: {json_output_directory}")
    print("\nNew structure created:")
    print(f"- Images named as 'counting_accuracy_0.png' through 'counting_accuracy_{num_prompts-1}.png'")
    print("- JSON file contains a single 'counting_accuracy' key with a list of all prompts")

if __name__ == "__main__":
    main()