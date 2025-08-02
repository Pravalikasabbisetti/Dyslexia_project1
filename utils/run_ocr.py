# utils/run_ocr.py
'''import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
from collections import defaultdict
import string
from jiwer import wer, cer

# ------------------------------
# 1. Character Mapping
# ------------------------------
CHARACTERS = string.ascii_letters + string.digits + " " + string.punctuation
char_to_idx = {c: i + 1 for i, c in enumerate(CHARACTERS)}  # 0 = blank
idx_to_char = {i: c for c, i in char_to_idx.items()}

# ------------------------------
# 2. CRNN Model
# ------------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.children())[:-3])
        self.rnn = nn.LSTM(input_size=256 * 8, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes + 1)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)

# ------------------------------
# 3. CTC Decode
# ------------------------------
def ctc_greedy_decode(log_probs):
    _, max_indices = log_probs.max(2)
    max_indices = max_indices.transpose(0, 1)
    results = []
    for indices in max_indices:
        pred = ""
        prev = -1
        for idx in indices:
            idx = idx.item()
            if idx != 0 and idx != prev:
                pred += idx_to_char.get(idx, "")
            prev = idx
        results.append(pred)
    return results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])'''

import os
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
def run_ocr(image_path):
    # Load pretrained OCR model
    model = ocr_predictor(pretrained=True)

    # Check if images were found
    if not image_path:
        print(" No image files found in the folder.")
    else:
        # Load images as a DocumentFile
        doc = DocumentFile.from_images(image_path)

        # Perform OCR
        result = model(doc)

        # Collect full text from all pages
        full_text = ""
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join([word.value for word in line.words])
                    full_text += line_text + " "

    return full_text