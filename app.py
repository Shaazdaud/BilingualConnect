import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from IndicTransTokenizer import IndicProcessor
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the two models and tokenizers
model_en_hi = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
tokenizer_en_hi = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)

model_hi_en = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True)
tokenizer_hi_en = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True)

ip = IndicProcessor(inference=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hindi_alphabets = "अआइईउऊऋऌएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशसहक्षज्ञ"

def translate_text(text, src_lang, tgt_lang):
    if src_lang == "eng_Latn" and tgt_lang == "hin_Deva":
        model = model_en_hi
        tokenizer = tokenizer_en_hi
    elif src_lang == "hin_Deva" and tgt_lang == "eng_Latn":
        model = model_hi_en
        tokenizer = tokenizer_hi_en
    else:
        raise ValueError("Unsupported language pair")

    batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)

    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

    translation = ip.postprocess_batch(generated_tokens, lang=tgt_lang)[0]
    return translation

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        # Detect the language of the input text
        if any(char in hindi_alphabets for char in user_input):
            src_lang = "hin_Deva"
            tgt_lang = "eng_Latn"
        else:
            src_lang = "eng_Latn"
            tgt_lang = "hin_Deva"

        translation = translate_text(user_input, src_lang, tgt_lang)
        return render_template("index.html", translation=translation)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)