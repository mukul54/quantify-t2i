import yaml
import argparse
import t2v_metrics.t2v_metrics as t2v_metrics
import json
import os
import numpy as np



def preprocess_texts(texts):
    processed_texts = []
    for text in texts:
        processed_texts.append(' '.join(text.split()[2:]))
    return processed_texts

def load_prompts(prompt_dir, category):
    # Load the JSON file
    with open(prompt_dir, 'r') as file:
        data = json.load(file)
    # Extract the list of prompts
    texts = data[category]

    # Processing the text
    processed_text = preprocess_texts(texts)
    return processed_text

def load_images(image_dir):
    filenames = sorted(os.listdir(image_dir), key= lambda x: int(x.split('_')[2].split('.')[0]))
    images = []
    for file in filenames:
        images.append(os.path.join(image_dir,file))
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    # Configuration
    with open(args.config, 'r') as file:
        data = yaml.safe_load(file)

    texts = load_prompts(data['prompt_dir'], data['category'])
    images = load_images(data['image_dir'])

    print('Evaluation Begins')
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl')
    print('Loaded Model')
    scores = clip_flant5_score(images=images, texts=texts)
    np.save(os.path.join(data['score_save_dir'], data['category']+'_scores.npy'), scores.cpu().numpy())