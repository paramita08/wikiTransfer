# Final actual code to match similarity of content and appending content:

import numpy as np
from pathlib import Path
import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('l3cube-pune/hindi-sentence-similarity-sbert')

# Update these paths to the absolute paths of your directories
input_dir_hin = os.path.expanduser('~/Desktop/Internship/FA_json_content_cleaned_hin/')
content_dir_trans = os.path.expanduser('~/Desktop/Internship/TESTING/filtered_translated_content_FA/')
output_dir = os.path.expanduser('~/Desktop/Internship/TESTING/hin_filtered_content_FA_Testing/')

# Check if input directories exist
if not os.path.exists(input_dir_hin):
    raise FileNotFoundError(f"Input directory not found: {input_dir_hin}")

if not os.path.exists(content_dir_trans):
    raise FileNotFoundError(f"Translated content directory not found: {content_dir_trans}")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_file_list_hin = [f for f in os.listdir(input_dir_hin) if os.path.isfile(os.path.join(input_dir_hin, f))]

for file in tqdm(input_file_list_hin, desc="Processing files"):
    qid = file.split('_')[0]
    file_path_hin = os.path.join(input_dir_hin, file)
    file_path_trans = os.path.join(content_dir_trans, f'{qid}.json')

    if not os.path.exists(file_path_trans):
        print(f"Translated file for QID {qid} not found.")
        continue

    try:
        with open(file_path_hin, 'r', encoding='utf-8') as f_hin, open(file_path_trans, 'r', encoding='utf-8') as f_trans:
            hin_content = json.load(f_hin)
            trans_content = json.load(f_trans)
    except Exception as e:
        print(f"Error loading JSON files for QID {qid}: {e}")
        continue

    new_dict = {}
    for key in hin_content.keys():
        if key in trans_content:
            pre_hin = hin_content[key].strip()
            trans_hin = trans_content[key].strip()

            if not pre_hin:
                # Directly use the translated value if the content is empty
                new_dict[key] = trans_hin
                #print(f"Empty content for QID {qid}, key {key}. Using translated value.")
                continue

            pre_sentences = sentence_tokenize.sentence_split(pre_hin, lang='hi')
            trans_sentences = sentence_tokenize.sentence_split(trans_hin, lang='hi')

            if len(pre_sentences) == 0 or len(trans_sentences) == 0:
                #print(f"No sentences found for QID {qid}, key {key}. Removing key.")
                continue

            try:
                pre_embeddings = model.encode(pre_sentences)
                trans_embeddings = model.encode(trans_sentences)
            except Exception as e:
                print(f"Error encoding sentences for QID {qid}, key {key}: {e}")
                continue

            indices = []
            for i in range(len(pre_sentences)):
                try:
                    if pre_embeddings.size == 0:
                        break
                except ValueError as e:
                    print(f"ValueError for QID {qid}, key {key}: {e}")
                    break

                try:
                    similarity_scores = cosine_similarity([pre_embeddings[i]], trans_embeddings).flatten()
                except Exception as e:
                    print(f"Error computing similarity for QID {qid}, key {key}, sentence {i}: {e}")
                    continue

                avg = np.mean(similarity_scores)
                std_dev = np.std(similarity_scores)
                indexed_values = list(enumerate(similarity_scores))
                sorted_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)
                filtered_values = [item for item in sorted_values if avg - std_dev <= item[1] <= avg + std_dev]
                sim_indices = [item[0] for item in filtered_values[:5]]
                indices.append(sim_indices)

            unique_indices = set(idx for row in indices for idx in row)
            final_str = pre_hin
            for idx in unique_indices:
                words = indic_tokenize.trivial_tokenize(trans_sentences[idx])
                if len(words) > 3:
                    final_str += trans_sentences[idx]

            new_dict[key] = final_str

    out_file = f'{qid}.json'
    out_path = os.path.join(output_dir, out_file)
    try:
        with open(out_path, 'w', encoding='utf-8') as json_file:
            json.dump(new_dict, json_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving JSON file for QID {qid}: {e}")
