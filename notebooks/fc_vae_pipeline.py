# easy_task_pipeline.py
"""
End-to-end pipeline for the 'Easy Task' of the VAE hybrid music clustering project.
- Extracts log-mel features from audio
- Optionally embeds lyrics (TF-IDF or sentence-transformers)
- Trains a simple fully-connected VAE on audio features
- Performs KMeans on VAE latents and PCA baseline
- Computes Silhouette Score and Calinski-Harabasz Index
- Visualizes with UMAP and t-SNE

Adjust DATA_DIR, hyperparams as needed.
"""
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

# --------------------------
# Config
# --------------------------
DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")     # audio files (.wav or .mp3)
LYRICS_CSV = os.path.join(DATA_DIR, "lyrics.csv")  # CSV with columns: id, filename, lyrics
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 30.0  # seconds (your clips are 30s)
USE_SENT_ENCODER = False  # if True, uses sentence-transformers for lyrics embeddings (optional)

# VAE params
INPUT_DIM = N_MELS * 2   # we'll use mean + std across time -> 2*N_MELS
HIDDEN_DIM = 512
LATENT_DIM = 32
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clustering params
N_CLUSTERS = 10  # adjust based on your dataset / expected genres/lang mix

# --------------------------
# Utilities: audio feature extraction
# --------------------------
def load_audio(path, sr=SAMPLE_RATE, duration=DURATION):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    # ensure length
    expected_len = int(sr * duration)
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))
    else:
        y = y[:expected_len]
    return y

def compute_log_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    logS = librosa.power_to_db(S, ref=np.max)
    return logS  # shape (n_mels, t_frames)

def audio_to_vector(path):
    y = load_audio(path)
    logmel = compute_log_mel(y)  # (n_mels, frames)
    mean = logmel.mean(axis=1)   # (n_mels,)
    std = logmel.std(axis=1)
    vec = np.concatenate([mean, std])  # (2*n_mels,)
    return vec.astype(np.float32)

# --------------------------
# Dataset class
# --------------------------
class AudioLyricDataset(Dataset):
    def __init__(self, items, audio_dir, lyric_embeddings=None):
        """
        items: list of dicts with keys: id, filename
        lyric_embeddings: dict id -> vector (optional)
        """
        self.items = items
        self.audio_dir = audio_dir
        self.lyric_embeddings = lyric_embeddings

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        path = os.path.join(self.audio_dir, it['filename'])
        audio_vec = audio_to_vector(path)  # shape (INPUT_DIM,)
        if self.lyric_embeddings is not None and it['id'] in self.lyric_embeddings:
            lyric_vec = self.lyric_embeddings[it['id']]
            return audio_vec, lyric_vec, it['id'], it['filename']
        else:
            return audio_vec, None, it['id'], it['filename']

# --------------------------
# Simple VAE (FC) for vector input
# --------------------------
class VAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)

        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec_fc(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE) + KL divergence
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld

# --------------------------
# Training function
# --------------------------
def train_vae(model, dataloader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0
        recon_sum = 0
        kld_sum = 0
        for batch in dataloader:
            audio_vecs = batch[0]  # shape (B, input_dim)
            audio_vecs = audio_vecs.to(DEVICE)
            opt.zero_grad()
            recon, mu, logvar = model(audio_vecs)
            loss, recon_l, kld = vae_loss(recon, audio_vecs, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            recon_sum += recon_l.item()
            kld_sum += kld.item()
        # print progress
        print(f"Epoch {epoch}/{epochs}  Loss={total_loss/len(dataloader.dataset):.4f}  Recon={recon_sum:.4f}  KLD={kld_sum:.4f}")
    return model

# --------------------------
# Helpers: lyrics embeddings (TF-IDF by default)
# --------------------------
def build_tfidf_embeddings(lyrics_df, id_col='id', text_col='lyrics', max_features=1024):
    corpus = lyrics_df[text_col].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus).toarray()
    ids = lyrics_df[id_col].tolist()
    emb = {ids[i]: X[i].astype(np.float32) for i in range(len(ids))}
    return emb, vectorizer

def build_sentence_transformer_embeddings(lyrics_df, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = lyrics_df['lyrics'].fillna("").astype(str).tolist()
    enc = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    ids = lyrics_df['id'].tolist()
    emb = {ids[i]: enc[i].astype(np.float32) for i in range(len(ids))}
    return emb

# --------------------------
# Main pipeline
# --------------------------
def main():
    # 1) Build item list from audio dir
    audio_files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(AUDIO_DIR, "*"))])
    # If you have a lyrics CSV mapping ids -> filename, use that. Otherwise create ids from filenames.
    if os.path.exists(LYRICS_CSV):
        lyrics_df = pd.read_csv(LYRICS_CSV)
        # Ensure columns id, filename, lyrics. If file paths are relative, keep filenames.
        items = []
        for _, row in lyrics_df.iterrows():
            items.append({'id': row['id'], 'filename': row['filename']})
    else:
        items = []
        for fn in audio_files:
            base = os.path.splitext(fn)[0]
            items.append({'id': base, 'filename': fn})

    print(f"Found {len(items)} items")

    # 2) Build lyrics embeddings (optional)
    lyric_embeddings = None
    if os.path.exists(LYRICS_CSV):
        lyrics_df = pd.read_csv(LYRICS_CSV)
        if USE_SENT_ENCODER:
            print("Building sentence-transformer embeddings for lyrics (this may be slow)...")
            lyric_embeddings = build_sentence_transformer_embeddings(lyrics_df)
        else:
            print("Building TF-IDF embeddings for lyrics...")
            lyric_embeddings, tf = build_tfidf_embeddings(lyrics_df)

    # 3) Create dataset and dataloader (we'll only use audio for VAE)
    # Precompute audio vectors to speed up training (cache)
    cache_file = os.path.join(DATA_DIR, "audio_vecs.pkl")
    if os.path.exists(cache_file):
        print("Loading cached audio vectors...")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
    else:
        print("Computing audio vectors (this can take time)...")
        cached = {}
        for it in tqdm(items):
            path = os.path.join(AUDIO_DIR, it['filename'])
            vec = audio_to_vector(path)
            cached[it['id']] = vec
        with open(cache_file, "wb") as f:
            pickle.dump(cached, f)

    # Build numpy dataset for VAE (only audio)
    all_ids = [it['id'] for it in items]
    X_audio = np.stack([cached[i] for i in all_ids], axis=0)  # (N, INPUT_DIM)
    print("Audio matrix shape:", X_audio.shape)

    # convert to torch dataset/dataloader
    tensor_data = torch.from_numpy(X_audio)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4) Build & train VAE
    model = VAE(input_dim=X_audio.shape[1], hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    print("Training VAE on audio features...")
    model = train_vae(model, dataloader, epochs=EPOCHS, lr=LR)

    # 5) Extract latent vectors (use mu)
    model.eval()
    latents = []
    with torch.no_grad():
        for i in range(0, X_audio.shape[0], BATCH_SIZE):
            batch = torch.from_numpy(X_audio[i:i+BATCH_SIZE]).to(DEVICE)
            mu, logvar = model.encode(batch)
            lat = mu.cpu().numpy()
            latents.append(lat)
    Z = np.vstack(latents)  # (N, LATENT_DIM)
    print("Latent shape:", Z.shape)

    # 6) Optionally augment with lyric embedding (concatenate)
    if lyric_embeddings is not None:
        print("Concatenating lyric embeddings (TF-IDF / sentence-transformer) to latents...")
        # Build matrix aligned to all_ids
        sample_lyric_vecs = []
        for idv in all_ids:
            if idv in lyric_embeddings:
                sample_lyric_vecs.append(lyric_embeddings[idv])
            else:
                # zero vector if missing
                sample_lyric_vecs.append(np.zeros_like(next(iter(lyric_embeddings.values()))))
        Ly = np.vstack(sample_lyric_vecs)  # (N, Ldim)
        print("Lyric embedding shape:", Ly.shape)
        # Optionally PCA lyric dims down if too large -- keep simple: normalize and concat
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Z_audio_scaled = sc.fit_transform(Z)
        Ly_scaled = sc.fit_transform(Ly)
        Z = np.concatenate([Z_audio_scaled, Ly_scaled], axis=1)
        print("Combined latent shape:", Z.shape)

    # 7) KMeans on VAE latents
    print("Running KMeans on VAE latents...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10).fit(Z)
    labels_vae = kmeans.labels_

    # 8) Baseline: PCA to latent_dim then KMeans
    print("Running PCA + KMeans baseline...")
    pca = PCA(n_components=LATENT_DIM, random_state=0)
    Z_pca = pca.fit_transform(X_audio)  # baseline using raw audio features -> PCA
    kmeans_pca = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10).fit(Z_pca)
    labels_pca = kmeans_pca.labels_

    # 9) Metrics
    print("Computing clustering metrics...")
    sil_vae = silhouette_score(Z, labels_vae)
    ch_vae = calinski_harabasz_score(Z, labels_vae)
    sil_pca = silhouette_score(Z_pca, labels_pca)
    ch_pca = calinski_harabasz_score(Z_pca, labels_pca)
    print("VAE+KMeans: Silhouette = {:.4f}, Calinski-Harabasz = {:.4f}".format(sil_vae, ch_vae))
    print("PCA+KMeans: Silhouette = {:.4f}, Calinski-Harabasz = {:.4f}".format(sil_pca, ch_pca))

    # 10) Visualize with UMAP and t-SNE
    print("Reducing to 2D with UMAP for visualization...")
    reducer = umap.UMAP(n_components=2, random_state=0)
    Z2d = reducer.fit_transform(Z)
    plt.figure(figsize=(8,6))
    plt.scatter(Z2d[:,0], Z2d[:,1], c=labels_vae, s=10, cmap='tab10')
    plt.title("VAE latent clusters (UMAP)")
    plt.savefig(os.path.join(DATA_DIR, "umap_vae_clusters.png"), dpi=150)
    plt.close()

    print("Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
    Z_tsne = tsne.fit_transform(Z)
    plt.figure(figsize=(8,6))
    plt.scatter(Z_tsne[:,0], Z_tsne[:,1], c=labels_vae, s=10, cmap='tab10')
    plt.title("VAE latent clusters (t-SNE)")
    plt.savefig(os.path.join(DATA_DIR, "tsne_vae_clusters.png"), dpi=150)
    plt.close()

    # Save results
    out = {
        'ids': all_ids,
        'Z': Z,
        'labels_vae': labels_vae,
        'labels_pca': labels_pca,
        'Z_pca': Z_pca,
        'sil_vae': sil_vae,
        'ch_vae': ch_vae,
        'sil_pca': sil_pca,
        'ch_pca': ch_pca
    }
    with open(os.path.join(DATA_DIR, "clustering_results.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("Saved clustering_results.pkl and visualizations under", DATA_DIR)

if __name__ == "__main__":
    main()
