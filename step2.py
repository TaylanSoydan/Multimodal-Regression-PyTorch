import pandas as pd
import numpy as np
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset
#from PIL import Image
from torchvision import transforms
from collections import Counter, OrderedDict
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.init as init
import random
import warnings
from torchvision import transforms
from torchvision.io import read_image
# Suppress only UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 80)

# For CPU and GPU
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# For Python's built-in random module (if used)
random.seed(42)

# For NumPy (if NumPy operations are used within PyTorch)
np.random.seed(42)

# Ensure that the program uses deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, desc_2_path, image_dir, vocab, max_len=5):
        """
        Args:
            dataframe (pd.DataFrame): Tabular dataframe containing features and the description column.
            desc_2_path (dict): Dictionary mapping descriptions to image file paths.
            image_dir (str): Directory where images are stored.
            vocab (Vocab): Vocabulary object to convert text to token indices.
            max_len (int): Maximum length of the tokenized text sequences.
        """
        self.dataframe = dataframe.drop(columns=['description'])  # Drop the 'description' column from tabular data
        self.descriptions = dataframe['description']  # Save the descriptions separately for tokenization
        self.desc_2_path = desc_2_path
        self.image_dir = image_dir
        self.vocab = vocab 
        self.max_len = max_len
        self.vocab_size = 93

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Tabular data
        tabular_data = torch.tensor(self.dataframe.iloc[idx].values, dtype=torch.float)

        # Text data 
        description = self.descriptions.iloc[idx].lower().split()  # Split text into words
        text_data = [self.vocab[word] if word in self.vocab else self.vocab['<unk>'] for word in description]
        text_data = text_data[:self.max_len]  # Truncate to max_len
        text_data = torch.tensor(text_data + [self.vocab['<pad>']] * (self.max_len - len(text_data)), dtype=torch.long)  # Pad sequence

        # Image data (load and transform the corresponding image)
        image_filename = self.desc_2_path.get(self.descriptions.iloc[idx])
        if image_filename is None:
            raise FileNotFoundError(f"No image found for description: {description}")

        image_path = os.path.join(self.image_dir, image_filename)
        image = transforms.ToPILImage()(read_image(image_path)).convert("RGB")
        #image = Image.open(image_path).convert('RGB')

        # Default image resizing and normalization for a simple CNN
        image = transforms.Resize((64, 64))(image) 
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        target = torch.tensor(self.dataframe.iloc[idx]['target'], dtype=torch.float) 

        return {'tabular': tabular_data,'text': text_data,'image': image,'target': target}

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MultimodalModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, cnn_out_dim, tabular_input_dim, tabular_hidden_dim, nonlinearity='relu'):
        """
        Args:
            vocab_size (int): Size of the vocabulary for text data.
            embedding_dim (int): Dimension of word embeddings for text data.
            hidden_dim (int): Hidden dimension for the RNN.
            cnn_out_dim (int): Output dimension of the CNN after the final convolutional layer.
            tabular_input_dim (int): Input dimension of the tabular data.
            tabular_hidden_dim (int): Hidden dimension for the tabular MLP.
            nonlinearity (str): Specifies the nonlinearity to use ('relu', 'leaky_relu', etc.).
        """
        super(MultimodalModel, self).__init__()

        self.nonlinearity = nonlinearity  # Store the nonlinearity type

        # Text (RNN) branch
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = 1, batch_first=True)
        self.rnn_layernorm = nn.LayerNorm(hidden_dim)

        # Image (CNN) branch
        x = 8
        self.cnn = nn.Sequential(
            nn.Conv2d(3, x, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(x),
            nn.ReLU() if nonlinearity == 'relu' else nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2),

            nn.Conv2d(x, 2 * x, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * x),
            nn.ReLU() if nonlinearity == 'relu' else nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(2 * x * 64 // 4 * 64 // 4, cnn_out_dim),
            nn.ReLU() if nonlinearity == 'relu' else nn.LeakyReLU(),
        )

        #Tabular (MLP) branch
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, tabular_hidden_dim),
            nn.ReLU() if nonlinearity == 'relu' else nn.LeakyReLU(),
        )

        # Fusion layer
        fusion_input_dim =  tabular_hidden_dim + cnn_out_dim + hidden_dim  
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 4),
            nn.ReLU() if nonlinearity == 'relu' else nn.LeakyReLU(),
            nn.Linear(4, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Proper initialization of the weights for convolutional layers, linear layers, 
        batch normalization layers, and embeddings in the network based on nonlinearity.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=1)

            elif isinstance(m, nn.GRU):
            # Initialize input-hidden weights
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity=self.nonlinearity)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, tabular_data, text_data, image_data):
        # Process text data through the RNN
        embedded_text = self.embedding(text_data)
        rnn_out, hn = self.rnn(embedded_text)
        rnn_out = rnn_out[:, -1, :] #rnn_out.mean(axis=1)
        rnn_out = self.rnn_layernorm(rnn_out)

        # Process image data through the CNN
        cnn_out = self.cnn(image_data)

        # Process tabular data through the MLP
        tabular_out = self.tabular_mlp(tabular_data)

        # Concatenate the outputs of the RNN, CNN, and MLP
        fused_output = torch.cat((rnn_out, cnn_out, tabular_out), dim=1)
        # Pass the fused output through the final regression head
        output = self.fusion_layer(fused_output)

        return output




def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, device, num_epochs=10, verbose=False, track_grads=False):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        all_targets = []
        all_predictions = []

        total_batches = len(train_dataloader)

        # Training loop
        for i_batch, batch in enumerate(train_dataloader):
            tabular_data = batch['tabular'].float().to(device)  # Tabular data
            text_data = batch['text'].to(device)  # Text data (tokenized as indices)
            image_data = batch['image'].to(device)  # Image data
            targets = batch['target'].float().to(device)  # Targets for regression

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(tabular_data, text_data, image_data).squeeze(1)
            outputs = torch.clamp(outputs, min=1)  # Ensure predictions are at least 1
            loss = criterion(outputs, targets)
            loss.backward()

            if track_grads:
                grad_norms_before_clipping, weights_before_clipping = compute_grad_norm_and_weights(model, track_weights=True)
                print(f"\nEpoch {epoch}, Grad Norms Before Clipping: {grad_norms_before_clipping}")
            # Clip gradients here
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100) 

            if track_grads:
                grad_norms_after_clipping, weights_after_clipping = compute_grad_norm_and_weights(model, track_weights=True)
                print(f"\nEpoch {epoch}, Grad Norms After Clipping: {grad_norms_after_clipping}")
                print(f"\nEpoch {epoch}, Weights After Clipping: {weights_after_clipping}")

            optimizer.step()

            running_loss += loss.item()  # Accumulate running loss inside the loop

            # Collect targets and predictions for R² score
            all_targets.extend(targets.detach().cpu().numpy())  # Original targets
            all_predictions.extend(outputs.detach().cpu().numpy())

            if verbose:
                percent_complete = (i_batch + 1) / total_batches * 100
                print(f"Epoch [{epoch+1}/{num_epochs}] - {percent_complete:.2f}% complete")
                train_r2 = r2_score(all_targets, all_predictions)
                train_rmse = root_mean_squared_error(all_targets, all_predictions)
                print(f"For training batches seen so far in this epoch, R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")  



        # Compute R² score for training data
        train_r2 = r2_score(all_targets, all_predictions)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training R²: {train_r2:.4f}, Training RMSE: {(running_loss / total_batches)**0.5:.4f}")  

        # Test Evaluation Phase
        model.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        all_test_targets = []
        all_test_predictions = []

        with torch.no_grad():
            for i_batch, batch in enumerate(test_dataloader):
                tabular_data = batch['tabular'].float().to(device)  # Tabular data
                text_data = batch['text'].to(device)  # Text data (tokenized as indices)
                image_data = batch['image'].to(device)  # Image data
                targets = batch['target'].float().to(device)  # Targets for regression

                # Forward pass
                outputs = model(tabular_data, text_data, image_data).squeeze(1)
                outputs = torch.clamp(outputs, min=1)  # Ensure predictions are at least 1
                loss = criterion(outputs, targets)

                test_loss += loss.item()

                # Collect targets and predictions for R² score
                all_test_targets.extend(targets.detach().cpu().numpy())  # Original targets
                all_test_predictions.extend(outputs.detach().cpu().numpy())

            # Compute R² score for test data
            test_r2 = r2_score(all_test_targets, all_test_predictions)
            print(f"Epoch [{epoch+1}/{num_epochs}], Test R²: {test_r2:.4f}, Test RMSE: {(test_loss/len(test_dataloader))**0.5:.4f}")  # Corrected RMSE

    print("Training complete.")
    return model


def build_vocab(descriptions, min_freq=1):
    """
    Builds a vocabulary from the text descriptions without using torchtext.
    
    Args:
        descriptions (pd.Series): A Pandas Series containing the text descriptions.
        min_freq (int): Minimum frequency for a word to be included in the vocabulary.
    
    Returns:
        vocab_dict (dict): A dictionary mapping words to indices.
        index_to_word (dict): A dictionary mapping indices to words (for reverse lookup).
        unk_index (int): Index of the '<unk>' token.
    """
    # Tokenize the descriptions
    tokenized_descriptions = [desc.lower().split() for desc in descriptions]

    # Flatten the list of tokenized descriptions and count word frequencies
    counter = Counter([word for desc in tokenized_descriptions for word in desc])

    # Filter words by frequency and sort by frequency
    sorted_by_freq = sorted([(word, freq) for word, freq in counter.items() if freq >= min_freq], 
                            key=lambda x: x[1], reverse=True)

    # Create an OrderedDict to maintain the order
    ordered_dict = OrderedDict(sorted_by_freq)

    # Add special tokens <pad> and <unk> manually (ensuring <pad> is 0, <unk> is 1)
    specials = ['<pad>', '<unk>']
    vocab_dict = {special: idx for idx, special in enumerate(specials)}
    index_to_word = {idx: special for special, idx in vocab_dict.items()}

    # Add regular words to vocab starting from len(specials)
    for idx, (word, _) in enumerate(ordered_dict.items(), start=len(specials)):
        vocab_dict[word] = idx
        index_to_word[idx] = word

    # Set the index for unknown tokens to '<unk>'
    unk_index = vocab_dict['<unk>']

    return vocab_dict, index_to_word, unk_index


# Path to the spacecraft_images directory
image_dir = "spacecraft_images"

# Path to the candidates_data.csv file
file = "candidates_data.csv"

df = pd.read_csv(file)
# Converting feature_1 and 2 to integers  
df["feature_1"] = df["feature_1"].str.split("_").str[1].astype(int)
df["feature_2"] = df["feature_2"].str.split("_").str[1].astype(int)

# Remove columns that consist of -5 and -2 
df = replace_special_values(df, epsilon=0.001, if_keep_bool=False)

# Drop missing features
print(f"Dropping columns with more than {20}% missing values")
df = df.drop(columns=df.columns[df.isnull().mean() > 0.01])

# Heuristically classify columns as numeric vs. categoric
column_classifications = classify_columns(df)
column_classifications["description"] = "Categorical"
cat_cols = [k for k,v in column_classifications.items() if v == "Categorical"]
num_cols = [k for k,v in column_classifications.items() if (v == "Numerical") and (k!="target")]

# Impute 
df = impute_data(df, num_cols=num_cols, cat_cols=cat_cols)

# Convert object columns to category 
object_columns = df.select_dtypes(include='object').columns
object_columns = object_columns[object_columns != 'description']
for col in object_columns:
    df[col] = df[col].astype('category').cat.codes

# Dict to map descriptions with image dirs
desc_2_path = dict(zip(list(df["description"].sort_values().unique()),sorted([file_name for file_name in os.listdir(image_dir)])))

#df = df.sample(frac=1, random_state=42)


print(df.shape)

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Store the 'description' column separately and drop it from both sets
train_descriptions = train_df.pop('description')
test_descriptions = test_df.pop('description')

# Separate the target column
y_train = train_df.pop('target')
y_test = test_df.pop('target')

# Initialize and fit the StandardScaler
scaler = StandardScaler()
train_df_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
test_df_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_df.index)

# Reattach 'description' and 'target' columns to the scaled data
train_df_scaled['description'] = train_descriptions
train_df_scaled['target'] = y_train
test_df_scaled['description'] = test_descriptions
test_df_scaled['target'] = y_test

# Build Vocab
descriptions = train_df_scaled['description']  
vocab_obj, index_to_word, unk_index = build_vocab(descriptions, min_freq=1)
# Dataloaders
train_dataset = MultimodalDataset(dataframe=train_df_scaled, 
                                  desc_2_path=desc_2_path, 
                                  image_dir=image_dir, 
                                  vocab=vocab_obj,
                                  max_len=5)
test_dataset = MultimodalDataset(dataframe=test_df_scaled, 
                                  desc_2_path=desc_2_path, 
                                  image_dir=image_dir, 
                                  vocab=vocab_obj,
                                  max_len=5)


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

# Training 
vocab_size = len(vocab_obj)  # Vocabulary size from the built vocab
embedding_dim = 1 
hidden_dim = 3 
cnn_out_dim = 20 
tabular_input_dim = train_dataset[0]['tabular'].shape[0] 
tabular_hidden_dim = 20 

model = MultimodalModel(vocab_size, embedding_dim, hidden_dim, cnn_out_dim, 
                        tabular_input_dim, tabular_hidden_dim, nonlinearity="leaky_relu")

# Print the number of trainable parameters in the multimodal model
print(f"Total number of trainable parameters: {count_parameters(model)}")
# Move the model to GPU if available
device = "cpu"
model = model.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam([
    {'params': model.cnn.parameters(), 'lr': 0.1},
    {'params': model.rnn.parameters(), 'lr': 1e-5},
    {'params': model.tabular_mlp.parameters(), 'lr': 0.001},  
], lr=0.001)  # Also slightly increase weight decay

print("Starting training the MultiModalModel")
# Training - Takes an hour on my laptop GPU
trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, 
                            device, num_epochs=1, verbose=True, track_grads=False)