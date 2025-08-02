import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import re
from spellchecker import SpellChecker
from metaphone import doublemetaphone
from nltk.corpus import words
import pandas as pd

# 1. EfficientNet extractor (freeze except last block)

class EfficientNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
        # Freeze all params
        for param in efficientnet.parameters():
            param.requires_grad = False
        # Unfreeze last conv block
        for param in efficientnet.features[-1].parameters():
            param.requires_grad = True

        self.backbone = efficientnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)  # (B, 1280)


# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. OCR feature extraction (unchanged from your code)

spell = SpellChecker()
try:
    english_words = set(words.words())
except:
    english_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

def get_phonetic_code(word):
    try:
        return doublemetaphone(word)[0] or word
    except:
        return word

def is_acronym(word):
    return word.isupper() and len(word) > 1

def is_proper_noun(word):
    return word[0].isupper() and word.lower() not in english_words

def extract_dyslexia_features(text):
 
    if text is None or pd.isna(text):
        text = ""

    # Convert to string and strip whitespace
    text = str(text).strip()

    # Initialize features dictionary
    features = {
        "Spelling errors": 0,
        "Unknown tokens": 0,
        "Incorrect capitalization": 0,
        "Missing capitalization": 0,
        "Phonetic errors": 0,
        "Letter transpositions": 0,
        "Omissions/additions": 0,
        "Punctuation errors": 0,
        "Inconsistent spacing": 0,
        "Repeated characters": 0
    }

    # If text is empty, return zero features
    if not text:
        return torch.tensor(list(features.values()), dtype=torch.float32)

    # Extract words and original words
    words_list = re.findall(r"\b\w+\b", text)
    original_words = text.split()

    if not words_list:  # No words found
        return torch.tensor(list(features.values()), dtype=torch.float32)

    # Basic text statistics for normalization
    total_words = len(words_list)
    total_chars = len(text)
    total_sentences = max(1, len(re.split(r'[.?!]\s*', text.strip())))

    # --- 1. Incorrect capitalization ---
    for i, word in enumerate(words_list):
        if len(word) > 0 and word[0].isupper() and i != 0 and not is_proper_noun(word) and not is_acronym(word):
            features["Incorrect capitalization"] += 1

    # --- 2. Missing capitalization (only sentence starts) ---
    sentences = re.split(r'[.?!]\s*', text.strip())
    for sent in sentences:
        if sent:
            sent_words = sent.strip().split()
            if sent_words and len(sent_words[0]) > 0 and sent_words[0][0].islower():
                features["Missing capitalization"] += 1

    # --- 3. Other word-level features ---
    for i, word in enumerate(words_list):
        if not word:  # Skip empty words
            continue

        word_lower = word.lower()

        try:
            correct = spell.correction(word_lower)
            if correct is None:
                correct = word_lower
        except:
            correct = word_lower

        # Spelling errors
        if correct != word_lower and word_lower not in english_words:
            features["Spelling errors"] += 1

        # Unknown tokens
        if word_lower not in english_words and len(spell.unknown([word_lower])) > 0:
            features["Unknown tokens"] += 1

        # Phonetic errors
        if correct != word_lower and get_phonetic_code(correct) == get_phonetic_code(word_lower):
            features["Phonetic errors"] += 1

        # Letter transpositions
        if correct != word_lower and sorted(correct) == sorted(word_lower):
            features["Letter transpositions"] += 1

        # Omissions/Additions
        if correct != word_lower:
            try:
                edit_dist = textdistance.levenshtein.distance(word_lower, correct)
                if edit_dist == 1:
                    features["Omissions/additions"] += 1
            except:
                pass

        # Repeated characters
        if re.search(r'(.)\1{2,}', word_lower):
            features["Repeated characters"] += 1

    # --- 4. Inconsistent spacing (split words) ---
    for i in range(len(original_words) - 1):
        if i + 1 < len(original_words):
            w1 = original_words[i].lower()
            w2 = original_words[i+1].lower()
            combined = w1 + w2
            if len(w1) <= 4 and len(w2) <= 4 and combined in english_words:
                features["Inconsistent spacing"] += 1

    # --- 5. Inconsistent spacing (glued words) ---
    for word in original_words:
        if not word:
            continue
        word_lower = word.lower()
        if word_lower not in english_words:
            for i in range(1, len(word_lower)):
                left = word_lower[:i]
                right = word_lower[i:]
                if left in english_words and right in english_words:
                    features["Inconsistent spacing"] += 1
                    break

    # --- 6. Punctuation errors ---
    punct_errors = re.findall(r"[^\w\s,.!?']", text)
    features["Punctuation errors"] += len(punct_errors)

    # Normalize features by text length/word count
    normalized_features = torch.tensor([
        features["Spelling errors"] / total_words,           # Spelling error rate
        features["Unknown tokens"] / total_words,            # Unknown token rate
        features["Incorrect capitalization"] / total_words,   # Incorrect capitalization rate
        features["Missing capitalization"] / total_sentences, # Missing capitalization rate per sentence
        features["Phonetic errors"] / total_words,           # Phonetic error rate
        features["Letter transpositions"] / total_words,     # Transposition rate
        features["Omissions/additions"] / total_words,       # Omission/addition rate
        features["Punctuation errors"] / total_chars,        # Punctuation error rate per character
        features["Inconsistent spacing"] / total_words,      # Spacing error rate
        features["Repeated characters"] / total_words        # Repeated character rate
    ], dtype=torch.float32)

    return normalized_features

# 4. Fusion Model (Balanced CNN + OCR)

class FusionClassifier(nn.Module):
    def __init__(self, cnn_dim=1280, ocr_dim=10):
        super().__init__()
        # CNN branch
        self.cnn_branch = nn.Sequential(
            nn.Linear(cnn_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # OCR branch
        self.ocr_branch = nn.Sequential(
            nn.Linear(ocr_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Fusion + final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, cnn_feat, ocr_feat):
        cnn_out = self.cnn_branch(cnn_feat)
        ocr_out = self.ocr_branch(ocr_feat)
        combined = torch.cat([cnn_out, ocr_out], dim=1)
        return self.classifier(combined)


# --- Prediction function ---
def predict_single_fusion(image_path, ocr_text, cnn_model_path, fusion_model_path, threshold=0.6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CNN feature extractor
    cnn_model = EfficientNetExtractor().to(device)
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.eval()

    # Load fusion classifier
    fusion_model = FusionClassifier().to(device)
    fusion_model.load_state_dict(torch.load(fusion_model_path, map_location=device))
    fusion_model.eval()

    # Preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Extract CNN features
    with torch.no_grad():
        cnn_feat = cnn_model(image_tensor)  # Shape (1,1280)

    # Extract OCR features (10-dim)
    ocr_feat = extract_dyslexia_features(ocr_text).unsqueeze(0).to(device)  # Shape (1,10)

    # Predict using fusion classifier
    with torch.no_grad():
        logits = fusion_model(cnn_feat, ocr_feat)  # Output (1,1)
        prob = torch.sigmoid(logits).item()
        prediction = 1 if prob > threshold else 0

    print(f"Predicted Probability: {prob:.4f}")
    print(f"Predicted Class: {'Dyslexic' if prediction == 1 else 'Non-Dyslexic'}")

    return prediction, prob