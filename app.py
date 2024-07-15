import torch
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import pickle
import torch.nn.functional as F
import torch.nn as nn

# Define your model architecture
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, feature_size=2048):
        super(CaptionModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + feature_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        features = features.unsqueeze(1).repeat(1, captions.size(1), 1)
        lstm_input = torch.cat((features, embeddings), dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        outputs = self.fc(lstm_out)
        return outputs


app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet model
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

# Load your caption model
model = CaptionModel(1848, 50, 256)
model.load_state_dict(torch.load(r'E:\imagecaption\saved_model\caption_model.pth', map_location=device))
model.to(device)
model.eval()

# Load word_to_idx and idx_to_word
with open(r'E:\imagecaption\storage\word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)
with open(r'E:\imagecaption\storage\idx_to_word.pkl', 'rb') as f:
    idx_to_word = pickle.load(f)

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_caption(img, beam_size=3):
    model.eval()
    max_len = 20
    with torch.no_grad():
        features = resnet(img.unsqueeze(0)).squeeze()
        start = torch.tensor([word_to_idx['startseq']]).unsqueeze(0).to(device)
        
        sequences = [(start, 0.0)]
        
        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                if seq[0, -1].item() == word_to_idx['endseq']:
                    all_candidates.append((seq, score))
                    continue
                
                output = model(features.unsqueeze(0), seq)
                output = output[:, -1, :]
                probabilities = F.softmax(output, dim=-1)
                
                top_probs, top_indices = probabilities.topk(beam_size)
                
                for i in range(beam_size):
                    next_seq = torch.cat([seq, top_indices[:, i].unsqueeze(1)], dim=1)
                    next_score = score - torch.log(top_probs[0, i]).item()
                    all_candidates.append((next_seq, next_score))
            
            sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_size]
            
            if all(seq[0, -1].item() == word_to_idx['endseq'] for seq, _ in sequences):
                break
    
    best_sequence = min(sequences, key=lambda x: x[1])[0].squeeze()
    caption = ' '.join([idx_to_word[idx.item()] for idx in best_sequence if idx.item() not in [word_to_idx['startseq'], word_to_idx['endseq']]])
    return caption
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        try:
            img = Image.open(file).convert('RGB')
            img = preprocess(img)
            img = img.to(device)
            caption = predict_caption(img)
            return render_template('result.html', caption=caption)
        except Exception as e:
            return f"An error occurred: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=False)