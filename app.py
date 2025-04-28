import numpy as np
import torch
import xgboost as xgb
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Load models
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
bert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
xgb_model = xgb.Booster()
xgb_model.load_model('model_fake_news_detector.json')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    headline = data.get('headline', '')
    body = data.get('body', '')
    
    # Combine headline and body
    text = headline + " " + body
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Extract BERT features
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            token_type_ids=encoding['token_type_ids']
        )
        # Get [CLS] token as text representation
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    # Predict with XGBoost
    dpredict = xgb.DMatrix(embeddings)
    prediction_proba = xgb_model.predict(dpredict)[0]
    prediction = "Valid" if prediction_proba > 0.5 else "Hoax"
    
    return jsonify({
        'prediction': prediction,
        'probability': float(prediction_proba),
        'label_numeric': 1 if prediction_proba > 0.5 else 0
    })

if __name__ == '__main__':
    app.run(debug=True)