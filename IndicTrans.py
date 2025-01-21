import sys
import json
import torch
import os
import math
from nltk.tokenize import sent_tokenize
from indicnlp.tokenize import sentence_tokenize
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransTokenizer import IndicTransTokenizer,IndicProcessor

tokenizer = IndicTransTokenizer(direction="indic-en")
#tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
ip = IndicProcessor(inference=True)
qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True, low_cpu_mem_usage=True,quantization_config=qconfig)
device = torch.device("cuda")

def translate_text(text):
  sentences = sentence_tokenize.sentence_split(text, lang='hi')
  translated_lines = []
  batch_size = 128

  for batch_id in range(0,math.ceil(len(sentences)/batch_size)):
      batch_sent = sentences[batch_id*batch_size:(batch_id + 1)*batch_size]
      batch = ip.preprocess_batch(batch_sent, src_lang="hin_Deva", tgt_lang="eng_Latn")
      model_inputs = tokenizer(batch, src=True, return_tensors="pt")
      model_inputs = model_inputs.to(device)
      with torch.inference_mode():
           outputs = model.generate(**model_inputs, num_beams=5, num_return_sequences=1, max_length=256)
           outputs = tokenizer.batch_decode(outputs, src=False)
           outputs = ip.postprocess_batch(outputs, lang="eng_Latn")
           translated_lines += outputs
           del model_inputs, outputs
           torch.cuda.empty_cache()

  translated_text = ""
  for line in translated_lines:
      translated_text += line
      #del model_inputs, generated_tokens, translation
      torch.cuda.empty_cache()
  #print(translated_text)
  return translated_text

cnt = 0
batch_size = 128

input_folder = os.path.expanduser('FA_Translation_Testing')
output_folder = os.path.expanduser('trans_FA')

json_files = [filename for filename in os.listdir(input_folder) if filename.endswith('.json')]

for json_filename in json_files:
    print(cnt)
    cnt += 1
    json_file_path = os.path.join(input_folder, json_filename)
    with open(json_file_path, 'r', encoding='utf-8') as file:
         data = json.load(file)
    
    #Translate values in each dictionary
    translated_data = {}
    for key, value in data.items():
        with torch.no_grad():
           hi_translations = translate_text(value)  # Translate the text value
           #print(key, hi_translations)
           #hi_translations = batch_translate(value, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)
           translated_data[key] = hi_translations


    out_file_name = json_filename.split('_')[0]
    out_file_name = out_file_name + '.json'
    #print(out_file_name)
    output_file_path = os.path.join(output_folder, out_file_name)
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
         json.dump(translated_data, output_file, ensure_ascii=False, indent=4)
      