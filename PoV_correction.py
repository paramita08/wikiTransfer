# BIAS REMOVAL FOR GA FILES :
import os
import json
import re
from tqdm import tqdm

# Define paths
input_folder_path = '/content/drive/MyDrive/Top_relevant_chunk_others'
output_folder_path = '/content/drive/MyDrive/Bias_removed_top_chunk_others'
quality_file_path = '/content/drive/MyDrive/Biography_quality_mapping/biography_quality_item_ids.json'

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Load the quality file
with open(quality_file_path, 'r', encoding='utf-8') as file:
    quality_data = json.load(file)

# Extract QIDs from the 'GA' category
ga_qids = set(quality_data.get('GA', []))

# Print the QIDs to ensure they are loaded correctly
print(f"GA QIDs: {ga_qids}")

# List all files in the input folder
all_files = [f for f in os.listdir(input_folder_path) if f.endswith('.json')]
# Print the total number of files found
print(f"Total files found: {len(all_files)}")

# Extract the QID part from each filename and filter the files to include only those in the 'GA' category
filtered_files = [f for f in all_files if f.split('_')[0] in ga_qids]

# Print the filtered files
print(f"Filtered files: {filtered_files}")

# Define the prompt for the API call
prompt_template = """For each query message, remove framing bias and epistemological bias and do not add any extra content from your own knowledge.

Framing bias is realized by subjective words or one-sided words, especially revealing the author’s stance in a particular debate.
Epistemological bias involves propositions that are either commonly agreed to be true or false and that are subtly presupposed, entailed, asserted or hedged in the text.

Here are some examples:
1. Message: Schnabel himself did the fantastic reproductions of Basquiat’s work.
   Response: Schnabel himself did the accurate reproductions of Basquiat’s work.

2. Message: Usually, smaller cottage-style houses have been demolished to make way for these McMansions.
   Response: Usually, smaller cottage-style houses have been demolished to make way for these homes.

3. Message: The first research revealed that the Meditation technique produces a unique state fact.
   Response: The first research indicated that the Meditation technique produces a unique state fact.

4. Message: Cooper says that slavery was worse in South America and the US than Canada, but clearly states that it was a horrible and cruel practice.
   Response: Cooper says that slavery was worse in South America and the US than Canada, but points out that it was a horrible and cruel practice.

5. Message: Colombian terrorist groups.
   Response: Colombian paramilitary groups.

Examples:
{}

I'll provide a section of text, and you should generate a neutral alternative considering the above-mentioned biases that avoids the author's opinions.
Provide only the Output as: <pad>output</pad>"""

# Function to process the content using the Groq API
def process_section(section_content, client):
    prompt = prompt_template.format(f'Section: {section_content}')
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": section_content,
            },
        ],
        model="llama3-70b-8192",
    )
    response_content = chat_completion.choices[0].message.content
    return response_content

# Iterate over each filtered JSON file in the input folder with tqdm to track progress
for filename in tqdm(filtered_files, desc="Processing files", unit="file"):
    input_file_path = os.path.join(input_folder_path, filename)
    output_file_path = os.path.join(output_folder_path, filename)
    
    # Read the JSON file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Process each section in the JSON
    for section, content in tqdm(data.items(), desc=f"Processing sections in {filename}", unit="section", leave=False):
        # Split content into sentences
        sentences = re.split(r'(?<=[.!?]) +', content)
        # Filter out sentences with less than 3 words
        filtered_sentences = [sentence for sentence in sentences if len(sentence.split()) >= 3]
        # Combine the filtered sentences into a single string
        filtered_content = ' '.join(filtered_sentences)
        
        # Process the whole filtered section using the API
        processed_content = process_section(filtered_content, client)
        data[section] = processed_content
    
    # Save the processed JSON to the output folder
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

print("Processing completed and files saved in 'Bias_removed_top_chunk_others' folder.")




# Define the folder containing the JSON files
folder_path = os.path.expanduser('~/Desktop/Internship/Voice_corrected_top_chunk_others')

# Iterate over each JSON file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('_top_chunk.json'):
        file_path = os.path.join(folder_path, filename)

        # Load the JSON data from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Iterate over each dictionary item in the JSON data
        for key, value in data.items():
            # Check if the value contains the specified string
            if 'Same sentence is already in third-person perspective, no changes needed.' in value:
                data[key] = ""

        # Save the modified JSON data back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

print("Processing complete.")