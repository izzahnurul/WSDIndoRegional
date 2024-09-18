import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, DistilBertTokenizer, DistilBertModel
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the original dataset and the augmented sentences dataset
datapath = ""  # Replace with preprocessed data path
df = pd.read_excel(datapath)
augmentedpath = ""  # Replace with augmented data path
augmented_df = pd.read_csv(augmentedpath)

# Load augmented data
augmented_df = pd.read_csv(augmented_data_path)

# Load real sentences
real_df = pd.read_excel(real_data_path)

# Ensure the data is in the right format
augmented_df = augmented_df.dropna(subset=['target_lan', 'ind', 'generated_sentence1', 'generated_sentence2', 'generated_sentence3', 'generated_sentence4', 'generated_sentence5'])

# Melt the augmented DataFrame to have each sentence as a separate row
melted_df = pd.melt(augmented_df, id_vars=['target_lan', 'ind'], value_vars=['generated_sentence1', 'generated_sentence2', 'generated_sentence3', 'generated_sentence4', 'generated_sentence5'],
                    var_name='sentence_type', value_name='sentence')
melted_df = melted_df.dropna(subset=['sentence'])  # Drop rows with missing sentences

# Merge real sentences with the augmented data to associate real sentences with the generated ones
merged_df = pd.merge(melted_df, real_df[['target_lan', 'ind', 'target_lan_p']], on=['target_lan', 'ind'], how='left')
merged_df = merged_df.dropna(subset=['target_lan_p'])  # Drop rows where there is no matching real sentence

# Initialize multiple models: BERT, RoBERTa, and DistilBERT
models = {
    'BERT': (BertTokenizer.from_pretrained('bert-base-uncased'), BertModel.from_pretrained('bert-base-uncased')),
    'RoBERTa': (RobertaTokenizer.from_pretrained('roberta-base'), RobertaModel.from_pretrained('roberta-base')),
    'DistilBERT': (DistilBertTokenizer.from_pretrained('distilbert-base-uncased'), DistilBertModel.from_pretrained('distilbert-base-uncased'))
}

# Function to get sentence-level embeddings using mean pooling
def get_sentence_embeddings(model, tokenizer, sentences, model_name="Model"):
    embeddings = []
    for sentence in tqdm(sentences, total=len(sentences), desc=f"Embedding with {model_name}"):
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean Pooling: Average over the token embeddings
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(sentence_embedding)

    return np.array(embeddings)

# Function to cluster embeddings and evaluate accuracy
def cluster_and_evaluate(model_name, real_embeddings, generated_embeddings, num_clusters=2):
    # Combine real and generated embeddings for clustering
    all_embeddings = np.vstack((generated_embeddings, real_embeddings))

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(all_embeddings)

    # Predict cluster assignments
    predictions = kmeans.predict(all_embeddings)

    # Labels: generated are 0 and real are 1
    real_labels = np.array([1] * len(real_embeddings))
    generated_labels = np.array([0] * len(generated_embeddings))
    all_labels = np.hstack((generated_labels, real_labels))

    # Calculate accuracy: how many real sentence embeddings fall into the same cluster as their generated sentence embeddings
    accuracy = accuracy_score(all_labels, predictions)
    print(f"Accuracy for {model_name}: {accuracy:.4f}")

    return kmeans, accuracy, predictions

# Function to plot t-SNE with a legend for generated and real data
def plot_tsne(embeddings, labels, predictions, title):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))

    # Plot real data points in blue and generated data points in red
    plt.scatter(tsne_embeddings[labels == 0, 0], tsne_embeddings[labels == 0, 1], c='red', label='Generated Data', s=5)
    plt.scatter(tsne_embeddings[labels == 1, 0], tsne_embeddings[labels == 1, 1], c='blue', label='Real Data', s=5)

    # Add a legend explaining the colors
    plt.legend(loc='upper right')

    plt.title(f"t-SNE Visualization of {title} Embeddings")
    plt.show()

# Iterate over models (BERT, RoBERTa, DistilBERT) to get sentence-level embeddings, cluster, evaluate, and plot t-SNE
for model_name, (tokenizer, model) in models.items():
    print(f"Processing {model_name}")

    # Get embeddings for generated sentences
    generated_embeddings = get_sentence_embeddings(model, tokenizer, merged_df['sentence'].tolist(), model_name=model_name)
    
    # Get embeddings for real sentences
    real_embeddings = get_sentence_embeddings(model, tokenizer, merged_df['target_lan_p'].tolist(), model_name=f"{model_name} (Real Sentences)")

    # Cluster, evaluate, and get predictions
    kmeans, accuracy, predictions = cluster_and_evaluate(model_name, real_embeddings, generated_embeddings)

    # Plot t-SNE for visualization
    all_embeddings = np.vstack((generated_embeddings, real_embeddings))
    all_labels = np.hstack((np.zeros(len(generated_embeddings)), np.ones(len(real_embeddings))))
    plot_tsne(all_embeddings, all_labels, predictions, f"{model_name}")
