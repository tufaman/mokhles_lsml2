from flask import Flask, request, render_template
from celery import Celery
from celery.result import AsyncResult
import numpy as np
import json
import pickle
import os
import re
import torch
from transformers import BertTokenizer, BertModel, logging
logging.set_verbosity_error()


celery_app = Celery('server', backend='redis://localhost', broker='redis://localhost')
app = Flask(__name__)


def load_model(pickle_path):
    with open(pickle_path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model('model.pkl')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.zero_grad()
bert_model.eval()

def encode_bert(heading, text, tokenizer, model):
  tokenized = tokenizer(text=heading, text_pair=text, return_tensors='pt',
                        max_length=128, truncation=True, truncation_strategy='only_second',
                        padding=True, pad_to_multiple_of=128)
  embedding = model(**tokenized)[1]
  return embedding.detach().numpy()

@celery_app.task
def predict(heading, text):
    with torch.no_grad():
        embedding = encode_bert(heading, text, bert_tokenizer, bert_model)
    result = model.predict(embedding)[0]
    if result:
        return "Positive"
    else:
        return "Negative"
    
@app.route('/')
def main_page():
    return "Hello, please use the predict handler"

@app.route('/predict', methods=["GET", "POST"])
def predict_handler():
    if request.method == 'POST':
        heading = request.form['heading']
        text = request.form['text']

        if not heading:
            flash('Heading is required!')
        elif not text:
            flash('Text is required!')
        else:
            task = predict(heading, text)
            return task
    return render_template('predict.html')

@app.route('/predict/<task_id>')
def predict_check_handler(task_id):
    task = AsyncResult(task_id, app=celery_app)
    if task.ready():
        response = {
            "status": "DONE",
            "result": task.result
        }
    else:
        response = {
            "status": "IN_PROGRESS"
        }
    return json.dumps(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
