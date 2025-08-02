# utils/classify_dyslexia.py
from utils.model import predict_single_fusion

def classify_dyslexia(image_path,ocr_text):
    cnn_model_path = "models/efficientnet_last_fold2_0.84.pth"
    fusion_model_path = "models/fusion_fold2_0.8462.pth"
    pred, prob = predict_single_fusion(image_path, ocr_text, cnn_model_path, fusion_model_path)
    return pred,prob