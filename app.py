from flask import Flask, render_template, request
from utils.classify_dyslexia import classify_dyslexia
import os
from werkzeug.utils import secure_filename
from utils.correct_text import corrected_text
from utils.run_ocr import run_ocr

app = Flask(__name__)
UPLOAD_FOLDER = 'images/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('image')
    if not file:
        return "No image uploaded", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    ocr_text =run_ocr(filepath)
    is_dyslexic , probability =classify_dyslexia(filepath, ocr_text)
    print(is_dyslexic)

    if is_dyslexic:
        correct_sent= corrected_text(ocr_text)
        print(correct_sent)
        corrected, suggestions, symptoms = (
            correct_sent["correct"],
            correct_sent["suggestions"],
            correct_sent["symptoms"]
        )

    else:
        corrected, symptoms, suggestions = "", [], []
    formatted_text = "\n".join(suggestions)
    return render_template('result.html',
                           image_path=filepath,
                           ocr_text=ocr_text,
                           symptoms = symptoms,
                           is_dyslexic=is_dyslexic,
                           probability = probability,
                           corrected =corrected,
                           suggested_text=formatted_text)

if __name__ == '__main__':
    app.run(debug=True,port=5002)

