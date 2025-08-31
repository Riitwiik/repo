import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pdfplumber
from PIL import Image
import pytesseract
from transformers import pipeline, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)


MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
device = 0 if torch.cuda.is_available() else -1  

summarizer = pipeline(
    "summarization",
    model=MODEL_NAME,
    device=device,
    torch_dtype=torch.float16 if device == 0 else None  
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_stream):
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            pg_text = page.extract_text()
            if pg_text:
                text += pg_text + "\n"
    return text.strip()

def extract_text_from_image(file_stream):
    image = Image.open(file_stream)
    text = pytesseract.image_to_string(image)
    return text.strip()

def split_text_by_tokens(text, max_tokens=800):
    """Split text into chunks by tokens"""
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        try:
            if ext == 'pdf':
                text = extract_text_from_pdf(file)
            else:
                text = extract_text_from_image(file)
        except Exception as e:
            return jsonify({"error": f"Failed to extract text: {str(e)}"}), 500
        return jsonify({"text": text}), 200
    return jsonify({"error": "Unsupported file format"}), 400

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get("text", "")
    length = data.get("length", "medium")

    if not text:
        return jsonify({"error": "No text to summarize"}), 400

    length_map = {
        "short": {"min": 50, "max": 100},
        "medium": {"min": 120, "max": 250},
        "long": {"min": 200, "max": 400}
    }
    chosen_length = length_map.get(length, length_map["medium"])

    chunks = split_text_by_tokens(text)

    
    try:
        results = summarizer(
            chunks,
            max_length=chosen_length["max"],
            min_length=chosen_length["min"] // 2,
            do_sample=False,
            batch_size=4  
        )
        summaries = [r['summary_text'] for r in results]
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

    
    if len(summaries) > 1:
        combined_summary = " ".join(summaries)
        try:
            result = summarizer(
                combined_summary,
                max_length=chosen_length["max"],
                min_length=chosen_length["min"],
                do_sample=False
            )
            summary = result[0]['summary_text']
        except Exception as e:
            return jsonify({"error": f"Final summarization failed: {str(e)}"}), 500
    else:
        summary = summaries[0] if summaries else ""

    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
