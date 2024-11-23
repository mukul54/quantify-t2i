import gradio as gr
import json
import os
import datetime
from PIL import Image
from pathlib import Path
import shutil
import tempfile

class ImageAnnotationApp:
    def __init__(self):
        # Load prompts from JSON
        base_dir = '/home/mukul.ranjan/Documents/diffusion/QuantifyT2I'
        self.prompts_file = os.path.join(base_dir, 'data/Prompts_final.json')
        self.updated_prompts_file = os.path.join(base_dir, 'data/Updated_Prompts.json')
        self.data_dir = Path(os.path.join(base_dir, 'data'))
        
        # Create temporary directory for images
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.temp_dir}")
        
        # Load original prompts
        with open(self.prompts_file, 'r') as f:
            self.prompts_data = json.load(f)
            
        # Load or create updated prompts file
        try:
            with open(self.updated_prompts_file, 'r') as f:
                self.updated_prompts = json.load(f)
        except FileNotFoundError:
            self.updated_prompts = {k: v.copy() for k, v in self.prompts_data.items()}
            with open(self.updated_prompts_file, 'w') as f:
                json.dump(self.updated_prompts, f, indent=4)
        
        # Initialize prompt categories
        self.categories = list(self.prompts_data.keys())
        
        # Create prompt mappings for dropdown
        self.prompt_mappings = {}
        for category in self.categories:
            self.prompt_mappings[category] = {
                f"{idx}: {prompt[:100]}...": (idx, prompt)
                for idx, prompt in enumerate(self.updated_prompts[category])
            }
        
        # Cache for image counts
        self.image_count_cache = {}
        
        # Add directory for selected images
        self.selected_images_dir = self.data_dir / 'Selected_Images'
        self.selected_images_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each category
        for category in self.categories:
            category_dir = self.selected_images_dir / category
            category_dir.mkdir(exist_ok=True)

    def get_image_path(self, category, prompt_idx, variation):
        """Get the correct image path with proper formatting"""
        base_path = self.data_dir / category
        
        # Special handling for Fractional_Representation
        if category == "Fractional_Representation":
            # For this category, there's only one image per prompt with different numbering
            image_name = f"fractional_representation_{prompt_idx + 1}.png"
            full_path = base_path / image_name
            print(f"Looking for Fractional image at: {full_path}")
            return str(full_path)
        
        # For other categories, use the standard pattern
        category_prefix_map = {
            "Counting_Accuracy": "counting",
            "Size_Proportionality": "size_proportionality",
            "Geometric_Shape_Understanding": "geometric_shape_understanding",
            "Numerical_Sequencing": "numerical_sequencing"
        }
        
        category_prefix = category_prefix_map.get(category, category.lower())
        image_name = f"{category_prefix}_{prompt_idx + 1}_{variation}.png"
        
        full_path = base_path / image_name
        print(f"Looking for image at: {full_path}")
        return str(full_path)

    def get_image_count(self, category, prompt_idx):
        """Get the number of available images for a prompt"""
        cache_key = f"{category}_{prompt_idx}"
        if cache_key in self.image_count_cache:
            return self.image_count_cache[cache_key]
        
        # Special handling for Fractional_Representation
        if category == "Fractional_Representation":
            image_path = self.get_image_path(category, prompt_idx, 1)  # variation doesn't matter here
            count = 1 if os.path.exists(image_path) else 0
            self.image_count_cache[cache_key] = count
            return count
        
        # For other categories, check up to 5 variations
        count = 0
        for i in range(1, 6):
            image_path = self.get_image_path(category, prompt_idx, i)
            if os.path.exists(image_path):
                count = i
            else:
                print(f"Missing image at: {image_path}")
        
        if count == 0:
            print(f"Warning: No images found for {category} prompt {prompt_idx + 1}")
        else:
            print(f"Total images found for {category} prompt {prompt_idx + 1}: {count}")
        
        self.image_count_cache[cache_key] = count
        return count

    def get_temp_image_path(self, original_path):
        """Copy image to temporary directory and return new path"""
        if not original_path or not os.path.exists(original_path):
            return None
            
        filename = os.path.basename(original_path)
        temp_path = os.path.join(self.temp_dir, filename)
        
        # Copy file if it hasn't been copied yet
        if not os.path.exists(temp_path):
            shutil.copy2(original_path, temp_path)
            
        return temp_path

    # def get_image_count(self, category, prompt_idx):
    #     """Get the number of available images for a prompt"""
    #     cache_key = f"{category}_{prompt_idx}"
    #     if cache_key in self.image_count_cache:
    #         return self.image_count_cache[cache_key]
        
    #     count = 0
    #     for i in range(1, 6):  # Check up to 5 images
    #         image_path = self.get_image_path(category, prompt_idx, i)
    #         if os.path.exists(image_path):
    #             count = i
    #             print(f"Found image {count} for {category} prompt {prompt_idx + 1}")
    #         else:
    #             print(f"Missing image at: {image_path}")
        
    #     if count == 0:
    #         print(f"Warning: No images found for {category} prompt {prompt_idx + 1}")
    #     else:
    #         print(f"Total images found for {category} prompt {prompt_idx + 1}: {count}")
        
    #     self.image_count_cache[cache_key] = count
    #     return count

    def load_images_for_prompt(self, category, prompt_idx):
        """Load all available images for a given prompt"""
        images = []
        image_count = self.get_image_count(category, prompt_idx)
        
        if category == "Fractional_Representation":
            # For this category, only load the single image if it exists
            image_path = self.get_image_path(category, prompt_idx, 1)  # variation doesn't matter
            if os.path.exists(image_path):
                temp_path = self.get_temp_image_path(image_path)
                images.append(temp_path)
            else:
                print(f"Missing image: {image_path}")
                images.append(None)
        else:
            # For other categories, load all variations
            for i in range(1, image_count + 1):
                image_path = self.get_image_path(category, prompt_idx, i)
                if os.path.exists(image_path):
                    temp_path = self.get_temp_image_path(image_path)
                    images.append(temp_path)
                else:
                    print(f"Missing image: {image_path}")
                    images.append(None)
        
        # Pad with None if less than 5 images
        while len(images) < 5:
            images.append(None)
        
        return images

    def get_prompts_for_category(self, category):
        """Get list of prompts for dropdown"""
        return list(self.prompt_mappings[category].keys())

    def get_prompt_info(self, category, prompt_selection):
        """Get prompt index and text from selection"""
        if not prompt_selection:
            return 0, self.updated_prompts[category][0]
        return self.prompt_mappings[category][prompt_selection]

    def update_prompt(self, category, prompt_text, prompt_idx):
        """Update prompt in the separate JSON file"""
        try:
            # Update the prompt in memory
            self.updated_prompts[category][prompt_idx] = prompt_text
            
            # Update prompt mappings
            self.prompt_mappings[category] = {
                f"{idx}: {prompt[:100]}...": (idx, prompt)
                for idx, prompt in enumerate(self.updated_prompts[category])
            }
            
            # Save to separate file
            with open(self.updated_prompts_file, 'w') as f:
                json.dump(self.updated_prompts, f, indent=4)
            
            # Create log entry
            log_entry = {
                "timestamp": str(datetime.datetime.now()),
                "category": category,
                "prompt_idx": prompt_idx,
                "old_prompt": self.prompts_data[category][prompt_idx],
                "new_prompt": prompt_text
            }
            
            # Append to log file
            log_file = os.path.join(os.path.dirname(self.prompts_file), 'prompt_updates_log.json')
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []
                
            logs.append(log_entry)
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=4)
                
            return f"Successfully updated prompt {prompt_idx} in {category}"
        except Exception as e:
            return f"Error updating prompt: {str(e)}"

    def save_annotation(self, category, prompt_idx, quality_rating, object_counts, notes):
        """Save annotation to a file"""
        annotation = {
            "category": category,
            "prompt_idx": prompt_idx,
            "prompt": self.updated_prompts[category][prompt_idx],
            "quality_rating": quality_rating,
            "object_counts": object_counts,
            "notes": notes,
            "timestamp": str(datetime.datetime.now())
        }
        
        annotations_file = os.path.join(os.path.dirname(self.prompts_file), 'annotations.json')
        try:
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            annotations = []
            
        annotations.append(annotation)
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=4)
        
        return "Annotation saved successfully!"

    def save_selected_images(self, category, prompt_idx, selected_indices):
        """Save selected images to the Selected_Images directory"""
        try:
            dest_dir = self.selected_images_dir / category
            
            # Create prompt directory
            prompt_dir = dest_dir / f"prompt_{prompt_idx + 1}"
            prompt_dir.mkdir(exist_ok=True)
            
            # Copy selected images
            for idx in selected_indices:
                src_path = self.get_image_path(category, prompt_idx, idx + 1)
                if os.path.exists(src_path):
                    dest_name = f"selected_{os.path.basename(src_path)}"
                    shutil.copy2(src_path, prompt_dir / dest_name)
            
            # Save selection metadata
            metadata = {
                "prompt": self.updated_prompts[category][prompt_idx],
                "selected_images": selected_indices,
                "timestamp": str(datetime.datetime.now())
            }
            
            with open(prompt_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
                
            return f"Successfully saved {len(selected_indices)} selected images for prompt {prompt_idx + 1}"
        except Exception as e:
            return f"Error saving selected images: {str(e)}"
    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks() as interface:
            with gr.Row():
                category_dropdown = gr.Dropdown(
                    choices=self.categories,
                    label="Category",
                    value=self.categories[0]
                )
                
            with gr.Row():
                prompt_dropdown = gr.Dropdown(
                    choices=self.get_prompts_for_category(self.categories[0]),
                    label="Select Prompt",
                    value=None
                )
                
            with gr.Row():
                prompt_textbox = gr.Textbox(label="Prompt", interactive=True)
                prompt_idx_display = gr.Number(label="Prompt Index", interactive=False)
                
            with gr.Row():
                update_prompt_btn = gr.Button("Update Prompt")
                
            with gr.Row():
                available_images_text = gr.Textbox(label="Available Images", interactive=False)
                
            with gr.Row():
                image_gallery = [gr.Image(label=f"Generation {i+1}") for i in range(5)]
                
            with gr.Row():
                image_checkboxes = [gr.Checkbox(label=f"Select Generation {i+1}", value=False) 
                                  for i in range(5)]
                
            with gr.Row():
                save_selection_btn = gr.Button("Save Selected Images")
                selection_status = gr.Textbox(label="Selection Status", interactive=False)
                
            with gr.Row():
                quality_rating = gr.Radio(
                    choices=["Poor", "Fair", "Good", "Excellent"],
                    label="Quality Rating"
                )
                
            with gr.Row():
                object_counts = gr.Textbox(
                    label="Object Counts (Format: object:expected:actual, comma-separated)",
                    placeholder="e.g., car:2:3, person:1:2"
                )
                
            with gr.Row():
                notes = gr.Textbox(
                    label="Additional Notes",
                    lines=3
                )
                
            with gr.Row():
                save_btn = gr.Button("Save Annotation")
                status_text = gr.Textbox(label="Status", interactive=False)
            
            def update_prompt_choices(category):
                prompts = self.get_prompts_for_category(category)
                return gr.Dropdown(choices=prompts)
            
            def load_prompt_and_images(category, prompt_selection):
                if not prompt_selection:
                    idx, prompt = 0, self.updated_prompts[category][0]
                else:
                    idx, prompt = self.get_prompt_info(category, prompt_selection)
                
                images = self.load_images_for_prompt(category, idx)
                image_count = self.get_image_count(category, idx)
                images_text = f"Found {image_count} images for this prompt"
                
                return {
                    prompt_textbox: prompt,
                    prompt_idx_display: idx,
                    available_images_text: images_text,
                    image_gallery[0]: images[0],
                    image_gallery[1]: images[1],
                    image_gallery[2]: images[2],
                    image_gallery[3]: images[3],
                    image_gallery[4]: images[4]
                }
            
            def save_image_selection(category, prompt_idx, *checkboxes):
                selected_indices = [i for i, checked in enumerate(checkboxes) if checked]
                result = self.save_selected_images(category, prompt_idx, selected_indices)
                return result
            
            def clear_selections(*args):
                return [False] * 5
            
            # Event handlers
            category_dropdown.change(
                update_prompt_choices,
                inputs=[category_dropdown],
                outputs=[prompt_dropdown]
            )
            
            prompt_dropdown.change(
                load_prompt_and_images,
                inputs=[category_dropdown, prompt_dropdown],
                outputs=[prompt_textbox, prompt_idx_display, available_images_text] + image_gallery
            )
            
            prompt_dropdown.change(
                clear_selections,
                inputs=[],
                outputs=image_checkboxes
            )
            
            update_prompt_btn.click(
                self.update_prompt,
                inputs=[category_dropdown, prompt_textbox, prompt_idx_display],
                outputs=[status_text]
            )
            
            save_selection_btn.click(
                save_image_selection,
                inputs=[category_dropdown, prompt_idx_display] + image_checkboxes,
                outputs=[selection_status]
            )
            
            save_btn.click(
                self.save_annotation,
                inputs=[
                    category_dropdown,
                    prompt_idx_display,
                    quality_rating,
                    object_counts,
                    notes
                ],
                outputs=[status_text]
            )
            
        return interface

    def __del__(self):
        """Cleanup temporary directory when done"""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")


if __name__ == "__main__":
    app = ImageAnnotationApp()
    interface = app.create_interface()
    
    # Create the Selected_Images directory structure
    base_dir = '/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data'
    selected_dir = os.path.join(base_dir, 'Selected_Images')
    os.makedirs(selected_dir, exist_ok=True)
    
    for category in app.categories:
        category_dir = os.path.join(selected_dir, category)
        os.makedirs(category_dir, exist_ok=True)
    
    try:
        interface.launch(
            share=True,
            allowed_paths=[
                "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data/Counting_Accuracy",
                "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data/Size_Proportionality",
                "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data/Fractional_Representation",
                "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data/Geometric_Shape_Understanding",
                "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data/Numerical_Sequencing",
                "/home/mukul.ranjan/Documents/diffusion/QuantifyT2I/data/Selected_Images"
            ]
        )
    finally:
        # Ensure cleanup happens even if there's an error
        if hasattr(app, 'temp_dir') and os.path.exists(app.temp_dir):
            shutil.rmtree(app.temp_dir)