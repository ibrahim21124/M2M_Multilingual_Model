#!/usr/bin/env python
# coding: utf-8

# Installing Dependencies
print("Welcome to CdiOneView Language Translator")
print("Initializing Dependencies")
# get_ipython().system('git clone https://github.com/pytorch/fairseq')

# get_ipython().system('pip install numpy')
# get_ipython().system('pip install pandas')
# get_ipython().system('pip install transformers')
# get_ipython().system('pip install sentencepiece')
# get_ipython().system('pip install flask')
# get_ipython().system('pip install fairseq')


import numpy as np
import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from flask import Flask, render_template, request, redirect, url_for, send_file
import os

os.system('wget -qq "https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model"')
os.system('wget -qq "https://dl.fbaipublicfiles.com/m2m_100/data_dict.128k.txt"')
os.system('wget -qq "https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt"')
os.system('wget -qq "https://dl.fbaipublicfiles.com/m2m_100/language_pairs_small_models.txt"')

print("Dependencies Installed! Loading Models")

# Output Generation

model_name = "M2M Model Pretrained by Meta"
model_repo = "facebook/m2m100_418M"
model = M2M100ForConditionalGeneration.from_pretrained(model_repo)
tokenizer = M2M100Tokenizer.from_pretrained(model_repo)
tokenizer.src_lang = "en"

def Generate_output(file, language):
    os.system(f"python3 fairseq/scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs={file} --outputs=english_text.en")
    os.system(f"fairseq-preprocess --seed=0 --source-lang en --target-lang {language} --only-source --testpref english_text --destdir data_bin --srcdict data_dict.128k.txt --tgtdict data_dict.128k.txt")
    os.system(f"fairseq-generate data_bin/ --seed=0 --batch-size 1 --path checkpoint_best.pt --fixed-dictionary model_dict.128k.txt -s en -t {language} --remove-bpe 'sentencepiece\' --beam 5 --task translation_multi_simple_epoch --lang-pairs language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset test > gen_out")
    os.system('grep ^H gen_out | cut -f3- > gen.out.sys')
    output_text = open("gen.out.sys", "r").read().split("\n")
    return output_text

print("Models Loaded! Initializing the Application")

# Web Application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title="Welcome to CdiOneView Language Translator")

@app.route('/translateexcel', methods=["GET", "POST"])
def upload():
    global destination
    
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        destination = '/'.join(["static/", uploaded_file.filename])
        uploaded_file.save(destination)
    else:
        return "No Input File Provided"
    
    input_dict = dict(request.form)
    print(input_dict)
    
    TO_LANGUAGE=str(list(input_dict.values())[0])
    with open(destination) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    input_text = [x.strip() for x in content]
    #     TEXT=pd.read_csv(destination, keep_default_na=False)
    OUTPUT_FILE_NAME = "OutputFile.csv"
    
    if(TO_LANGUAGE == "de"):
        LANGUAGE="German"
    elif(TO_LANGUAGE == "es"):
        LANGUAGE="Spanish"
    elif(TO_LANGUAGE == "ja"):
        LANGUAGE="Japanese"
    elif(TO_LANGUAGE == "fr"):
        LANGUAGE="French"
    elif(TO_LANGUAGE == "it"):
        LANGUAGE="Italian"
    elif(TO_LANGUAGE == "da"):
        LANGUAGE="Danish"
    elif(TO_LANGUAGE == "nl"):
        LANGUAGE="Dutch"
    elif(TO_LANGUAGE == "sv"):
        LANGUAGE="Swedish"
        
#     input_text = TEXT["English"]
#     output_text = []
    
#     for text in input_text:
#       text_token = tokenizer_trained(text, padding=True, max_length = 128,truncation=True, return_tensors="pt")
#       output_text_generate = model_trained.generate(**text_token, forced_bos_token_id=tokenizer_trained.get_lang_id(TO_LANGUAGE))
#       output_text.append(tokenizer_trained.batch_decode(output_text_generate, skip_special_tokens=True))

    output_text = Generate_output(destination, TO_LANGUAGE)
    output_text_final = output_text[:-1]
    
    output_dataset = {LANGUAGE : output_text_final}
    output_df = pd.DataFrame(output_dataset)
    output_df.to_csv(OUTPUT_FILE_NAME)
    return_output_df = output_df.head(n=10)
    
    return render_template("translateexcel.html", output_file=OUTPUT_FILE_NAME, columns=return_output_df.columns, data=return_output_df)

@app.route('/translate', methods=["POST"])
def translate():
    input_dict = dict(request.form)
    print(input_dict)
    
    TEXT=list(input_dict.values())[0]
    TO_LANGUAGE=str(list(input_dict.values())[1])
    
    if(TO_LANGUAGE == "de"):
        LANGUAGE="German"
    elif(TO_LANGUAGE == "es"):
        LANGUAGE="Spanish"
    elif(TO_LANGUAGE == "ja"):
        LANGUAGE="Japanese"
    elif(TO_LANGUAGE == "fr"):
        LANGUAGE="French"
    elif(TO_LANGUAGE == "it"):
        LANGUAGE="Italian"
    elif(TO_LANGUAGE == "da"):
        LANGUAGE="Danish"
    elif(TO_LANGUAGE == "nl"):
        LANGUAGE="Dutch"
    elif(TO_LANGUAGE == "sv"):
        LANGUAGE="Swedish"
    
    input_text = TEXT
    
    # M2M Output Generation
    text_token = tokenizer(input_text, padding=True, max_length = 128,truncation=True, return_tensors="pt")
    output_text_generate = model.generate(**text_token, forced_bos_token_id=tokenizer.get_lang_id(TO_LANGUAGE))
    output_text = tokenizer.batch_decode(output_text_generate, skip_special_tokens=True)

    TRANS=output_text[0]
    
    return render_template('translate.html', text=TEXT, trans=TRANS, language=LANGUAGE)

@app.route('/basicversion', methods=["GET"])
def translateupload():
    return render_template("basicversion.html")

@app.route('/download/<filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

app.run()
