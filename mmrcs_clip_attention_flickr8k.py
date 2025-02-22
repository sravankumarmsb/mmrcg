# Encoder CLIP - Decoder ATTEN with 30k Training Data

# Importin Libraries
import pickle
import clip

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import re
import spacy
import configparser

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read("config.ini")

# Read base path from config file
base_path = config.get("paths", "base_path_8k")


class Vocabulary:
  print("Debug: 1.1")
  itos = {}
  stoi = {}
  spacy_eng = spacy.load('en_core_web_sm')
  def __init__(self, token_file):
    print("Debug: 1.2")
    df = pd.read_csv(token_file)
    for idx, row in df.iterrows():
      key = int(row['Index'])
      val = str(row['Tokens']).strip()
      self.itos[key] = val
      self.stoi[val] = key

  def __len__(self):
    print("Debug: 1.3")
    return len(self.itos)

  def tokenize(self, text):
    print("Debug: 1.4")
    return [token.text.lower() for token in self.spacy_eng.tokenizer(text)]

  def numericalize(self,text):
    print("Debug: 1.5")
    tokenized_text = self.tokenize(text)
    return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
  
class CaptionUtil:
  print("Debug: 2.1")
  caption_dict = {}
  def __init__(self, image_folder, captions_file, tokens_file, train_file, test_file, val_file):
    print("Debug: 2.2")
    self.captionsDF = pd.read_csv(captions_file)
    self.captionsDF['caption'] = self.captionsDF['caption'].apply(lambda sentence: re.sub('[^a-zA-Z ]+', '', sentence.lower().strip()))
    self.captionsDF['token_count'] = self.captionsDF['caption'].apply(lambda sentence: len(sentence.split()))
    self.optimal_sentence_length = int(np.round(np.average(self.captionsDF['token_count']) + np.std(self.captionsDF['token_count'])*2))
    self.train_test_split(train_file, test_file, val_file)
    self.vocab = Vocabulary(tokens_file)
    for image in self.val_images + self.test_images:
      self.caption_dict[image] = self.captionsDF[self.captionsDF['image'] == image]['caption'].tolist()

  def read2list(self, filename):
    print("Debug: 2.3")
    f = open(filename, 'r')
    data = f.read()
    l = data.split('\n')
    if(l[len(l)-1] == ""):
      l.remove("")
    f.close()
    return l

  def train_test_split(self, train_file, test_file, val_file):
    print("Debug: 2.4")
    self.train_images = self.read2list(train_file)
    self.test_images = self.read2list(test_file)
    self.val_images = self.read2list(val_file)
    self.all_images = self.train_images + self.test_images + self.val_images

  def get_train_set(self):
    print("Debug: 2.5")
    return self.captionsDF[self.captionsDF['image'].isin(self.train_images)]

class ImageUtil:
  print("Debug: 3.1")
  @staticmethod
  def transform(image_path, image_name):
    print("Debug: 3.2")
    transform_image = T.Compose([T.Resize((224,224)), T.ToTensor()])
    img = Image.open(os.path.join(image_path,image_name)).convert("RGB")
    return transform_image(img)

  @staticmethod
  def show_image(inp, title=None):
    print("Debug: 3.3")
    """Imshow for Tensor"""
    inp = inp.numpy().transpose((1,2,0))
    plt.imshow(inp)
    if title is not None:
      plt.title(title)
    plt.pause(0.001)

image_embedding_file = os.path.join(base_path, "Extracts", "clip_vit_b32_embeddings.pkl")

print("DEBUG: 4.1")
with open(image_embedding_file, 'rb') as f:
  clip_embeddings = pickle.load(f)
print("DEBUG: 4.2")

class CustomDataset(Dataset):
  print("Debug: 5.1")
  def __init__(self,root_dir, traindf, vocab):
    print("Debug: 5.2")
    self.root_dir = root_dir
    self.df = traindf
    self.imgs = self.df["image"].tolist()
    self.captions = self.df["caption"].tolist()
    self.vocab = vocab

  def __len__(self):
    print("Debug: 5.3")
    return len(self.df)

  def __getitem__(self,idx):
    print("Debug: 5.4")
    caption = self.captions[idx]
    img_name = self.imgs[idx]
    img = clip_embeddings[img_name]
    caption_vec = []
    caption_vec += [self.vocab.stoi["<SOS>"]]
    caption_vec += self.vocab.numericalize(caption)
    caption_vec += [self.vocab.stoi["<EOS>"]]
    return img, torch.tensor(caption_vec)

class CapsCollate:
  print("Debug: 6.1")
  def __init__(self,pad_idx,batch_first=False):
    print("Debug: 6.2")
    self.pad_idx = pad_idx
    self.batch_first = batch_first

  def __call__(self,batch):
    print("Debug: 6.3")
    imgs = [item[0] for item in batch]
    imgs = torch.cat(imgs,dim=0)
    targets = [item[1] for item in batch]
    targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
    return imgs,targets    
  
class DecoderRNN(nn.Module):
  print("Debug: 7.1")
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, num_heads=2):
    print("Debug: 7.2")
    super(DecoderRNN,self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.vocab_size = vocab_size

    # Additional MLP layers for embedding transformation
    # Embedding layer to convert token indices to dense vectors
    self.embedding = nn.Embedding(vocab_size, embed_size)
    # Additional MLP layers for embedding transformation
    self.mlp_emb = nn.Sequential(nn.Linear(embed_size, embed_size),
                                  nn.LayerNorm(embed_size),
                                  nn.ELU(),
                                  nn.Linear(embed_size, embed_size))
    # LSTM layer for sequential processing
    self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=True)
    # Multi-head attention mechanism to capture dependencies between tokens
    self.attention = nn.MultiheadAttention(embed_dim=hidden_size,
                                            num_heads=num_heads,
                                            batch_first=True,
                                            dropout=0.1)
    # Final MLP layers for output transformation
    self.mlp_out = nn.Sequential(nn.Linear(hidden_size, hidden_size//2),
                                  nn.LayerNorm(hidden_size//2),
                                  nn.ELU(),
                                  nn.Dropout(0.5),
                                  nn.Linear(hidden_size//2, vocab_size))

  def forward(self, captions, hidden_seq, hidden_in, mem_in, features, cum_caption_embeds):
    print("Debug: 7.3")
    # Convert input tokens to dense vectors using embedding layer
    input_embs = self.embedding(captions)
    if(cum_caption_embeds.shape[0] == 0):
      cum_caption_embeds = input_embs
    else:
      cum_caption_embeds = cum_caption_embeds + input_embs

    # Additional MLP layers for embedding transformation
    input_embs = self.mlp_emb(input_embs)
    input_embs = features.unsqueeze(1) + input_embs
    #input_embs = torch.cat((features.unsqueeze(1),input_embs),dim=1)
    # Pass input embeddings through LSTM layer
    output, (hidden_out, mem_out) = self.lstm(input_embs, (hidden_in, mem_in))
    # Log the output of the final LSTM layer
    hidden_seq += [output]
    hidden_cat = torch.cat(hidden_seq, 1)
    # Apply multi-head attention mechanism over LSTM outputs
    # Use a single query from the current timestep
    # Keys and Values created from the outputs of LSTM from all previous timesteps
    attn_output, attn_output_weights = self.attention(output, hidden_cat, hidden_cat)  # Q, K, V
    # Combine attention output with LSTM output
    attn_output = attn_output + output
    # Apply final MLP layers for output transformation
    return self.mlp_out(attn_output), hidden_seq, hidden_out, mem_out,cum_caption_embeds

  def init_hidden(self, batch_size):
    return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(torch.float32)
    
def generate_caption(model, image,max_len=20,vocab=None):
  hidden_seq = []
  hiddens = model.init_hidden(image.shape[0]).to(device)
  memory = model.init_hidden(image.shape[0]).to(device)
  caption_text = []
  cum_caption_embeds = torch.tensor([]).to(device)
  captions = torch.tensor([[vocab.stoi['<SOS>']]]).to(device)
  for i in range(max_len):
    output, hidden_seq, hiddens, memory, cum_caption_embeds = model(captions, hidden_seq, hiddens,memory, image, cum_caption_embeds)
    predicted_word_idx = output.view(-1, model.vocab_size).argmax(dim=1)
    captions = torch.tensor([[predicted_word_idx.item()]]).to(device)
    try:
      token = vocab.itos[predicted_word_idx.item()]
      if token == "<EOS>":
        break
    except:
      token = '<UNK>'
    if not (token == '<SOS>' or token == '<EOS>'):
      caption_text.append(token)
  return ' '.join(caption_text)

class EncoderDecoder(nn.Module):
  print("Debug: 8.1")
  def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1,drop_prob=0.3):
    print("Debug: 8.2")
    super(EncoderDecoder,self).__init__()
    self.decoder = DecoderRNN(embed_size,hidden_size,vocab_size,num_layers,drop_prob)

  def forward(self, images, captions):
    print("Debug: 8.3")
    outputs = self.decoder(images, captions)
    return outputs
  
# Caption Directories
print("Debug: 4.3")
image_data_location = base_path

captions_file = os.path.join(base_path, "captions.csv")
token_file = os.path.join(base_path, "Extracts", "tokens.csv")
train_image_csv = os.path.join(base_path, "Extracts", "train_images.txt")
test_image_csv = os.path.join(base_path, "Extracts", "test_images.txt")
val_image_csv = os.path.join(base_path, "Extracts", "val_images.txt")

print("Debug: 4.4")
Caption_Reader = CaptionUtil(image_data_location, captions_file, token_file, train_image_csv, test_image_csv, val_image_csv)
pad_idx = Caption_Reader.vocab.stoi["<PAD>"]
dataset = CustomDataset(root_dir = image_data_location, traindf = Caption_Reader.get_train_set(), vocab = Caption_Reader.vocab)
# data_loader = DataLoader(dataset=dataset, batch_size=50, num_workers=1, shuffle=True, collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True))
# data_loader = DataLoader(dataset=dataset, batch_size=300, num_workers=4, shuffle=True, collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Debug: 4.5")
# initialize model, loss etc
# model = EncoderDecoder(embed_size = 512, hidden_size = 512, vocab_size = len(Caption_Reader.vocab), num_layers=3).to(device)
model = DecoderRNN(embed_size = 512, hidden_size = 512, vocab_size = len(Caption_Reader.vocab), num_layers=3, num_heads=8).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Function to Load Checkpoint
def load_ckp(checkpoint_path, model, optimizer):
    print("Debug: 8.1")
    # checkpoint = torch.load(checkpoint_path)
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def test_epoch(epoch_num, test_images, model, optimizer, criterion, Caption_Reader, image_data_location, image_embeddings):
  print("Debug: 9.1")
  image = []
  prediction = []
  model.eval()
  with torch.no_grad():
    for i in range(len(test_images)):
      img = image_embeddings[test_images[i]].to(torch.float32).to(device)
      prediction.append(generate_caption(model, img,vocab=Caption_Reader.vocab, max_len=Caption_Reader.optimal_sentence_length))
      image.append(test_images[i])
  return image, prediction

def clip_embedding(image):
    print("Debug: 10.1")
    import skimage.io as io
    # image_embeddings = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load('ViT-B/32', device=device, jit=False)

    image = io.imread(image)
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).cpu()
        # image_embeddings[img_name] = prefix
    return prefix

def imageToCaptions(image):
    print("Debug: 11.1")
    print("Predicting ")
    # initialize model, loss etc
    model = DecoderRNN(embed_size = 512, hidden_size = 512, vocab_size = len(Caption_Reader.vocab), num_layers=3, num_heads=8).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Debug: 11.2")
    checkpoint_path = os.path.join(base_path, "clip_attn_checkpoint_20.pt")
    model, optimizer, loaded_epoch = load_ckp(checkpoint_path, model, optimizer)
    model.eval()
    print("Debug: 11.3")
    with torch.no_grad():
        print("image: ", image)
        img = clip_embedding(image).to(torch.float32).to(device)
        print("Clip Embeddings: ", img)
        prediction = generate_caption(model,img,max_len=Caption_Reader.optimal_sentence_length,vocab=Caption_Reader.vocab)
        print("Prediction: ", prediction)
    return prediction
    