# FOR FA FILES :
# Top 3 section header matching with similarity scores :


# #Translate and get similarity between two section headings for all files in FA:
import os
import json
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransTokenizer import IndicProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def translate_keys(input_folder, hin_filename, model_name):
    # Load the Hindi JSON content
    hin_json_file = os.path.join(input_folder, hin_filename)
    with open(hin_json_file, 'r', encoding='utf-8') as hin_file:
        hin_data = json.load(hin_file)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    ip = IndicProcessor(inference=True)
    src_lang, tgt_lang = "hin_Deva", "eng_Latn"

    # Preprocess the keys for translation
    keys = list(hin_data.keys())
    batch = ip.preprocess_batch(keys, src_lang=src_lang, tgt_lang=tgt_lang)

    # Tokenize and encode the keys
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")
    DEVICE = "cpu"
    inputs = inputs.to(DEVICE)

    # Generate translations using the model
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens into English text
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess the translations
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    # Return a dictionary of Hindi keys and their translations
    return {keys[i]: translations[i] for i in range(len(keys))}

def load_json_keys(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return list(data.keys())

def get_embeddings(keys, model):
    return model.encode(keys)

# Initialize the embedding model
#embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #(MODEL REJECTED DUE TO LOW SCORES)
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2')


# Define input and output directories
input_folder_en = os.path.expanduser('~/Desktop/Internship/FA_json_content_cleaned_en')
input_folder_hin = os.path.expanduser('~/Desktop/Internship/FA_json_content_cleaned_hin')
output_folder = os.path.expanduser('~/Desktop/Internship/New_FA_sec_map_json')

# Ensure output folder exists, create if not
os.makedirs(output_folder, exist_ok=True)

# Get list of Hindi JSON files
hin_files = [filename for filename in os.listdir(input_folder_hin) if filename.endswith('.json')]

# Process each Hindi JSON file separately
for hin_filename in tqdm(hin_files, desc="Processing Files"):
    # QID from the Hindi filename (assuming format is QID_hin_content.json)
    QID = hin_filename.split('_')[0]

    # English JSON file for current QID
    filename_en = f"{QID}_content.json"
    file_path_en = os.path.join(input_folder_en, filename_en)
    if not os.path.exists(file_path_en):
        continue

    keys_en = load_json_keys(file_path_en)

    # Hindi JSON file for current QID
    file_path_hin = os.path.join(input_folder_hin, hin_filename)

    # Translate Hindi keys to English
    model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
    translated_dict = translate_keys(input_folder_hin, hin_filename, model_name)

    # Extract original Hindi keys and their translated English keys
    keys_hin = list(translated_dict.keys())
    translated_keys_en = list(translated_dict.values())

    # Generate embeddings for both sets of keys
    embeddings_en = get_embeddings(keys_en, embedding_model)
    embeddings_trans = get_embeddings(translated_keys_en, embedding_model)

    # Convert embeddings to numpy arrays for similarity calculation
    embeddings_en = np.array(embeddings_en)
    embeddings_trans = np.array(embeddings_trans)

    # Calculate cosine similarity between each pair of English and translated keys
    similarity_matrix = cosine_similarity(embeddings_en, embeddings_trans)

    # Create a dictionary to store top 3 similarities for each en_key
    similarity_dict = {}

    # Iterate over each English key and find top 3 similarities
    for i, en_key in enumerate(keys_en):
        similarity_scores = similarity_matrix[i]
        top_indices = np.argsort(similarity_scores)[::-1][:3]  # Indices of top 3 similarities
        top_similarities = [(keys_hin[idx], float(similarity_scores[idx])) for idx in top_indices]
        similarity_dict[en_key] = top_similarities

    # Define output file path for JSON
    output_file_path = os.path.join(output_folder, f"{QID}_sec_map.json")

    # Write similarities to a JSON file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(similarity_dict, output_file, ensure_ascii=False, indent=4)

    #tqdm.write(f"Similarity output for {QID} saved to {output_file_path}")

tqdm.write("Processing completed.")




# FOR OTHERS :

#Translate and get similarity between two section headings for all files in GA, A, B, C:
import os
import json
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransTokenizer import IndicProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def translate_keys(input_folder, hin_filename, model_name):
    # Load the Hindi JSON content
    hin_json_file = os.path.join(input_folder, hin_filename)
    with open(hin_json_file, 'r', encoding='utf-8') as hin_file:
        hin_data = json.load(hin_file)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    ip = IndicProcessor(inference=True)
    src_lang, tgt_lang = "hin_Deva", "eng_Latn"

    # Preprocess the keys for translation
    keys = list(hin_data.keys())
    batch = ip.preprocess_batch(keys, src_lang=src_lang, tgt_lang=tgt_lang)

    # Tokenize and encode the keys
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")
    DEVICE = "cpu"
    inputs = inputs.to(DEVICE)

    # Generate translations using the model
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens into English text
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess the translations
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    # Return a dictionary of Hindi keys and their translations
    return {keys[i]: translations[i] for i in range(len(keys))}

def load_json_keys(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return list(data.keys())

def get_embeddings(keys, model):
    return model.encode(keys)

# Initialize the embedding model
#embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  #(MODEL REJECTED DUE TO LOW SCORES)
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2')

# Define input and output directories
input_folder_en = os.path.expanduser('~/Desktop/Internship/en_json_content_cleaned')
input_folder_hin = os.path.expanduser('~/Desktop/Internship/hin_json_content_cleaned')
output_folder = os.path.expanduser('~/Desktop/Internship/New_sec_map_json')

# Ensure output folder exists, create if not
os.makedirs(output_folder, exist_ok=True)

# Get list of Hindi JSON files
hin_files = [filename for filename in os.listdir(input_folder_hin) if filename.endswith('.json')]

# Process each Hindi JSON file separately
for hin_filename in tqdm(hin_files, desc="Processing Files"):
    # QID from the Hindi filename (assuming format is QID_hin_content.json)
    QID = hin_filename.split('_')[0]

    # English JSON file for current QID
    filename_en = f"{QID}_content.json"
    file_path_en = os.path.join(input_folder_en, filename_en)
    if not os.path.exists(file_path_en):
        continue

    keys_en = load_json_keys(file_path_en)

    # Hindi JSON file for current QID
    file_path_hin = os.path.join(input_folder_hin, hin_filename)

    # Translate Hindi keys to English
    model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
    translated_dict = translate_keys(input_folder_hin, hin_filename, model_name)

    # Extract original Hindi keys and their translated English keys
    keys_hin = list(translated_dict.keys())
    translated_keys_en = list(translated_dict.values())

    # Generate embeddings for both sets of keys
    embeddings_en = get_embeddings(keys_en, embedding_model)
    embeddings_trans = get_embeddings(translated_keys_en, embedding_model)

    # Convert embeddings to numpy arrays for similarity calculation
    embeddings_en = np.array(embeddings_en)
    embeddings_trans = np.array(embeddings_trans)

    # Calculate cosine similarity between each pair of English and translated keys
    similarity_matrix = cosine_similarity(embeddings_en, embeddings_trans)

    # Create a dictionary to store top 3 similarities for each en_key
    similarity_dict = {}

    # Iterate over each English key and find top 3 similarities
    for i, en_key in enumerate(keys_en):
        similarity_scores = similarity_matrix[i]
        top_indices = np.argsort(similarity_scores)[::-1][:3]  # Indices of top 3 similarities
        top_similarities = [(keys_hin[idx], float(similarity_scores[idx])) for idx in top_indices]
        similarity_dict[en_key] = top_similarities

    # Define output file path for JSON
    output_file_path = os.path.join(output_folder, f"{QID}_sec_map.json")

    # Write similarities to a JSON file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(similarity_dict, output_file, ensure_ascii=False, indent=4)

    #tqdm.write(f"Similarity output for {QID} saved to {output_file_path}")

tqdm.write("Processing completed.")
