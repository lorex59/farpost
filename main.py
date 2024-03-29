from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# Данный код можно запустить с помощью команды uvicorn main:app --reload

app = FastAPI()

path_to_model = "ai-forever/RuM2M100-418M" # Путь к модели на hugging Face

model = M2M100ForConditionalGeneration.from_pretrained(path_to_model)  # Загрузка модели
tokenizer = M2M100Tokenizer.from_pretrained(path_to_model, src_lang="ru", tgt_lang="ru") # Токенизатор

class Form(BaseModel): # Принимающая форма
    sentence: str

class Prediction(BaseModel):
    sentence: str
    correct_sentence: str
    
    
@app.get('/status')
def status():
    print("test")
    return 'ok'
    
@app.post("/predict", response_model=Prediction)
def predict(form: Form):
    sentence = form.model_dump()["sentence"]
    encodings = tokenizer(sentence, return_tensors="pt") 
    generated_tokens = model.generate(
            **encodings, forced_bos_token_id=tokenizer.get_lang_id("ru"))
    answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return {
        'sentence': sentence,
        'correct_sentence': answer
    }
    
    