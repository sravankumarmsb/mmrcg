# Encoder CLIP - Decoder LSTM with 8k Training Data

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
  def __init__(self,embed_size,hidden_size,vocab_size,num_layers=1,drop_prob=0.3):
    print("Debug: 7.2")
    super(DecoderRNN,self).__init__()
    self.embedding = nn.Embedding(vocab_size,embed_size)
    self.lstm = nn.LSTM(embed_size,hidden_size,num_layers=num_layers,batch_first=True)
    self.fcn = nn.Linear(hidden_size,vocab_size)
    self.drop = nn.Dropout(drop_prob)

  def forward(self,features, captions):
    print("Debug: 7.3")
    embeds = self.embedding(captions[:,:-1])
    x = torch.cat((features.unsqueeze(1),embeds),dim=1)
    x,_ = self.lstm(x)
    x = self.fcn(x)
    return x

  def generate_caption(self,inputs,hidden=None,max_len=20,vocab=None):
    print("Debug: 7.4")
    batch_size = inputs.size(0)
    captions = []
    for i in range(max_len):
      output,hidden = self.lstm(inputs,hidden)
      output = self.fcn(output)
      output = output.view(batch_size,-1)
      predicted_word_idx = output.argmax(dim=1)
      inputs = self.embedding(predicted_word_idx.unsqueeze(0))

      try:
        token = vocab.itos[predicted_word_idx.item()]
        if token == "<EOS>":
          break
      except:
        token = '<UNK>'
      if not (token == '<SOS>' or token == '<EOS>'):
        captions.append(token)

    return ' '.join(captions)


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Debug: 4.5")
# initialize model, loss etc
model = EncoderDecoder(embed_size = 512, hidden_size = 512, vocab_size = len(Caption_Reader.vocab), num_layers=3).to(device)
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
      img = image_embeddings[test_images[i]].unsqueeze(0).to(torch.float32).to(device)
      prediction.append(model.decoder.generate_caption(img,vocab=Caption_Reader.vocab, max_len=Caption_Reader.optimal_sentence_length))
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
    model = EncoderDecoder(embed_size = 512, hidden_size = 512, vocab_size = len(Caption_Reader.vocab), num_layers=3).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Debug: 11.2")
    checkpoint_path = os.path.join(base_path, "clip_lstm_checkpoint_20.pt")
    model, optimizer, loaded_epoch = load_ckp(checkpoint_path, model, optimizer)
    model.eval()
    print("Debug: 11.3")
    with torch.no_grad():
        print("image: ", image)
        # img = clip_embeddings[image].unsqueeze(0).to(torch.float32).to(device)
        img = clip_embedding(image).unsqueeze(0).to(torch.float32).to(device)
        print("Clip Embeddings: ", img)
        prediction = model.decoder.generate_caption(img,vocab=Caption_Reader.vocab, max_len=Caption_Reader.optimal_sentence_length)
        print("Prediction: ", prediction)
    return prediction
    