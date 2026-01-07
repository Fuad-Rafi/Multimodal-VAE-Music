# medium_task_pipeline.py
"""
Medium Task pipeline (Conv-VAE + Multi-modal fusion + clustering + evaluation + visualization)
Implements all 5 Medium Task points from the project spec. Outputs saved under ./results/
Author: Generated for your project
References: Project Medium Task spec. :contentReference[oaicite:2]{index=2}
"""

import os, glob, random, math, pickle, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score,
                             calinski_harabasz_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.manifold import TSNE

# ---------- CONFIG ----------
DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
LYRICS_CSV = os.path.join(DATA_DIR, "lyrics.csv")     # optional
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")     # optional (for ARI/NMI)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Audio processing
SR = 22050
DURATION = 30.0               # seconds
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Training
BATCH_SIZE = 32
EPOCHS_VAE = 10  # Reduced from 60 for faster testing
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model dims
LATENT_DIM = 64               # VAE latent
LYRIC_PCA_DIM = 32            # reduced lyric dim for concatenation
FUSION_DIM = LATENT_DIM + LYRIC_PCA_DIM

# Clustering
N_CLUSTERS = 10               # baseline; we will vary when appropriate

# Other
USE_SENT_TRANSFORMERS = False   # set True to use sentence-transformers for lyrics (optional)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip VAE training if checkpoint exists")
    parser.add_argument("--epochs", type=int, default=EPOCHS_VAE, help="Number of VAE training epochs")
    return parser.parse_args()

# ---------- UTIL: audio -> mel spectrogram (saved as numpy) ----------
def load_audio(path, sr=SR, duration=DURATION):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    expected = int(sr * duration)
    if len(y) < expected:
        y = np.pad(y, (0, expected - len(y)))
    else:
        y = y[:expected]
    return y

def mel_spectrogram(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    logS = librosa.power_to_db(S, ref=np.max)
    return logS            # shape (n_mels, time_frames)

def precompute_mels(audio_dir=AUDIO_DIR, out_dir=os.path.join(DATA_DIR,"mels")):
    os.makedirs(out_dir, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(audio_dir,"*")))
    cache_map = {}
    for p in tqdm(audio_files, desc="Precompute mels"):
        fn = os.path.splitext(os.path.basename(p))[0]
        outp = os.path.join(out_dir, fn + ".npy")
        if os.path.exists(outp):
            cache_map[fn] = outp
            continue
        y = load_audio(p)
        m = mel_spectrogram(y)
        # normalize per-file (zscore)
        m = (m - m.mean()) / (m.std() + 1e-9)
        np.save(outp, m.astype(np.float32))
        cache_map[fn] = outp
    return cache_map

# ---------- Dataset ----------
class MelLyricDataset(Dataset):
    def __init__(self, items, mel_map, lyric_embeddings=None, transform=None):
        """
        items: list of dicts {'id','filename'}
        mel_map: dict id -> mel .npy path
        lyric_embeddings: dict id -> vector (optional)
        """
        self.items = items
        self.mel_map = mel_map
        self.lyric_embeddings = lyric_embeddings
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        idd = it['id']
        mel = np.load(self.mel_map[idd])   # (n_mels, t)
        # For conv nets, we want shape (1, n_mels, t)
        mel = mel[np.newaxis, :, :]
        lyric = None
        if self.lyric_embeddings is not None and idd in self.lyric_embeddings:
            lyric = self.lyric_embeddings[idd]
        sample = {'id': idd, 'mel': mel, 'lyric': lyric, 'filename': it['filename']}
        if self.transform:
            sample = self.transform(sample)
        return sample

# ---------- Conv VAE ----------
class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=LATENT_DIM):
        super().__init__()
        # Encoder: [1,128,T] -> conv down to small map
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # -> 32 x 64 x T/2
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # -> 64 x 32 x T/4
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),         # -> 128 x 16 x T/8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),        # -> 256 x 8 x T/16
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        # We'll infer flattened size during first forward
        self._flattened = None
        self.fc_mu = None
        self.fc_logvar = None
        self.fc_dec = None
        self.decoder = None
        self.latent_dim = latent_dim

    def _init_linear_heads(self, x):
        # x is feature map after enc (B, C, H, W)
        B, C, H, W = x.shape
        self._flattened = C * H * W
        self.fc_mu = nn.Linear(self._flattened, self.latent_dim).to(DEVICE)
        self.fc_logvar = nn.Linear(self._flattened, self.latent_dim).to(DEVICE)
        self.fc_dec = nn.Linear(self.latent_dim, self._flattened).to(DEVICE)
        # decoder from reshaped feature map to original spectrogram size using ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(C, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # final: may overshoot time dimension; we will crop in forward if necessary
        ).to(DEVICE)
        # move newly created heads to device (already done)
    
    def encode(self, x):
        h = self.enc(x)
        if self._flattened is None:
            # initialize heads
            self._init_linear_heads(h)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar, h.shape

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, shape):
        # shape = (B,C,H,W) to reshape dec output
        x = self.fc_dec(z)
        x = x.view(-1, shape[1], shape[2], shape[3])
        x = self.decoder(x)
        # ensure final number of channels is 1 and shape matches input approximate
        return x

    def forward(self, x):
        mu, logvar, shape = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.decode(z, shape)
        # crop recon to input width if necessary
        # if recon.shape != x.shape: crop/pad in time dimension
        if recon.shape[2:] != x.shape[2:]:
            # align along height/time dims by cropping center region
            _,_,h1,w1 = recon.shape
            _,_,h0,w0 = x.shape
            h_start = max(0, (h1 - h0)//2)
            w_start = max(0, (w1 - w0)//2)
            recon = recon[:, :, h_start:h_start+h0, w_start:w_start+w0]
        return recon, mu, logvar

# ---------- Loss ----------
def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld

# ---------- Lyrics embeddings ----------
def build_tfidf(lyrics_csv=LYRICS_CSV, max_features=2048):
    df = pd.read_csv(lyrics_csv)
    texts = df['lyrics'].fillna("").astype(str).tolist()
    ids = df['id'].astype(str).tolist()
    vec = TfidfVectorizer(max_features=max_features)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    emb = {ids[i]: X[i] for i in range(len(ids))}
    return emb, vec

def build_sentence_transformer(lyrics_csv=LYRICS_CSV, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    df = pd.read_csv(lyrics_csv)
    texts = df['lyrics'].fillna("").astype(str).tolist()
    ids = df['id'].astype(str).tolist()
    model = SentenceTransformer(model_name)
    X = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    emb = {ids[i]: X[i] for i in range(len(ids))}
    return emb

# ---------- Helper: load items list ----------
def build_items(audio_dir=AUDIO_DIR, lyrics_csv=LYRICS_CSV):
    audio_files = sorted(glob.glob(os.path.join(audio_dir,"*")))
    items = []
    if os.path.exists(lyrics_csv):
        lyrics_df = pd.read_csv(lyrics_csv)
        # ensure filenames correspond
        for _,r in lyrics_df.iterrows():
            items.append({'id': str(r['id']), 'filename': str(r['filename'])})
    else:
        for p in audio_files:
            fn = os.path.splitext(os.path.basename(p))[0]
            items.append({'id': fn, 'filename': os.path.basename(p)})
    return items

# ---------- Training loop ----------
def train_vae(model, dataloader, epochs=EPOCHS_VAE, lr=LR, device=DEVICE):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(1,epochs+1):
        model.train()
        total_loss = 0
        for batch in dataloader:
            xs = torch.stack([torch.from_numpy(s['mel']).float() for s in batch]).to(device)  # (B,1,H,W)
            opt.zero_grad()
            recon, mu, logvar = model(xs)
            loss, r, k = vae_loss(recon, xs, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(dataloader.dataset)
        if epoch % 5 == 0 or epoch == 1 or epoch==epochs:
            print(f"Epoch {epoch}/{epochs} avg_loss={avg:.4f}")
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"convvae_epoch{epoch}.pt"))
    return model

# ---------- Extract latents ----------
def extract_latents(model, dataset):
    model.eval()
    # use same collate as training to avoid PyTorch default collate errors when lyrics is None
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
    ids = []
    latents = []
    with torch.no_grad():
        for batch in loader:
            xs = torch.stack([torch.from_numpy(s['mel']).float() for s in batch]).to(DEVICE)
            _, mu, _ = model(xs)
            lat = mu.cpu().numpy()
            latents.append(lat)
            ids.extend([s['id'] for s in batch])
    Z = np.vstack(latents)
    return ids, Z

# ---------- Baselines ----------
def pca_baseline(mel_map, items, pca_dim=LATENT_DIM):
    # flatten mel into vector mean+std concatenation (fast baseline), then PCA
    X = []
    ids = []
    for it in items:
        m = np.load(mel_map[it['id']])
        feat = np.concatenate([m.mean(axis=1), m.std(axis=1)])   # 2*N_MELS
        X.append(feat)
        ids.append(it['id'])
    X = np.vstack(X)
    pca = PCA(n_components=pca_dim, random_state=RANDOM_SEED)
    Zp = pca.fit_transform(X)
    return ids, Zp

# ---------- Clustering & evaluation ----------
def run_cluster_and_eval(Z, ids, labels_true_dict=None, method_name="KMeans", params=None, n_clusters=N_CLUSTERS):
    metrics = {}
    if method_name == "KMeans":
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(Z)
    elif method_name == "Agglomerative":
        km = AgglomerativeClustering(n_clusters=n_clusters)
        labels = km.fit_predict(Z)
    elif method_name == "DBSCAN":
        eps = params.get('eps', 1.0)
        min_samples = params.get('min_samples', 5)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(Z)
    else:
        raise ValueError("Unknown method")

    # mask for single cluster or noise-only
    unique = np.unique(labels)
    if len(unique) <= 1:
        # degenerate clustering; set metrics to NaN
        metrics['silhouette'] = float('nan')
        metrics['davies_bouldin'] = float('nan')
        metrics['calinski_harabasz'] = float('nan')
    else:
        metrics['silhouette'] = silhouette_score(Z, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(Z, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(Z, labels)

    # ARI / NMI if labels present
    if labels_true_dict:
        true_labels = [labels_true_dict[i] if i in labels_true_dict else -1 for i in ids]
        # only compute ARI/NMI on samples with valid labels
        mask = np.array([t != -1 for t in true_labels])
        if mask.sum() > 1:
            metrics['ARI'] = adjusted_rand_score(np.array(true_labels)[mask], np.array(labels)[mask])
            metrics['NMI'] = normalized_mutual_info_score(np.array(true_labels)[mask], np.array(labels)[mask])
        else:
            metrics['ARI'] = float('nan')
            metrics['NMI'] = float('nan')

    metrics['n_clusters_found'] = int(len(unique))
    metrics['labels'] = labels
    return metrics

# ---------- Visualization ----------
def embed_and_plot(Z, labels, ids, tag, out_prefix):
    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
    Z2 = reducer.fit_transform(Z)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=Z2[:,0], y=Z2[:,1], hue=labels, palette='tab10', legend=None, s=8)
    plt.title(f"UMAP: {tag}")
    plt.savefig(f"{out_prefix}_umap_{tag}.png", dpi=150)
    plt.close()
    # t-SNE (slower)
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, init='pca', learning_rate='auto')
    Zt = tsne.fit_transform(Z)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=Zt[:,0], y=Zt[:,1], hue=labels, palette='tab10', legend=None, s=8)
    plt.title(f"t-SNE: {tag}")
    plt.savefig(f"{out_prefix}_tsne_{tag}.png", dpi=150)
    plt.close()
    return Z2

# ---------- Reconstruct and save spectrogram images ----------
def save_reconstructions(model, dataset, out_prefix, n=8):
    model.eval()
    # pick n random items
    idxs = np.random.choice(len(dataset), size=min(n,len(dataset)), replace=False)
    fig, axes = plt.subplots(2, len(idxs), figsize=(3*len(idxs), 6))
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            s = dataset[idx]
            x = torch.from_numpy(s['mel']).unsqueeze(0).float().to(DEVICE)  # (1,1,H,W)
            recon, _, _ = model(x)
            x_np = x.cpu().numpy()[0,0]
            recon_np = recon.cpu().numpy()[0,0]
            axs = axes[:, i]
            librosa.display.specshow(x_np, sr=SR, hop_length=HOP_LENGTH, y_axis='mel', x_axis='time', ax=axs[0])
            axs[0].set_title("Original")
            librosa.display.specshow(recon_np, sr=SR, hop_length=HOP_LENGTH, y_axis='mel', x_axis='time', ax=axs[1])
            axs[1].set_title("Reconstruction")
    plt.tight_layout()
    plt.savefig(out_prefix + "_reconstructions.png", dpi=150)
    plt.close()

# ---------- Main ----------
def main():
    args = parse_args()
    epochs = args.epochs
    print("=" * 80)
    print("MEDIUM TASK PIPELINE - STARTING")
    print("=" * 80)
    print("1) Precompute mel spectrograms")
    mel_map = precompute_mels()

    print("2) Build items list")
    items = build_items()
    print("Total items:", len(items))
    # keep only items that have a precomputed mel
    items_all = items
    items = [it for it in items_all if it['id'] in mel_map]
    if len(items) != len(items_all):
        print(f"Filtered items to those with mels: {len(items)} kept, {len(items_all) - len(items)} dropped")

    # optionally load labels
    labels_true = {}
    if os.path.exists(LABELS_CSV):
        labdf = pd.read_csv(LABELS_CSV)
        for _,r in labdf.iterrows():
            labels_true[str(r['id'])] = r['label']

    # Lyrics embeddings
    lyric_embeddings = None
    if os.path.exists(LYRICS_CSV):
        print("Building lyric embeddings (TF-IDF)...")
        if USE_SENT_TRANSFORMERS:
            lyric_embeddings = build_sentence_transformer()
        else:
            lyric_embeddings, tf = build_tfidf()
        print("Built lyric embeddings for", len(lyric_embeddings), "items")

        # Reduce lyric dim with PCA for stable concatenation
        ids_with_lyrics = [it['id'] for it in items if it['id'] in lyric_embeddings]
        Ly = np.vstack([lyric_embeddings[i] for i in ids_with_lyrics])
        pca = PCA(n_components=LYRIC_PCA_DIM, random_state=RANDOM_SEED)
        Ly_reduced = pca.fit_transform(Ly)
        # create new dict reduced
        lyric_emb_reduced = {}
        for i, idx in enumerate(ids_with_lyrics):
            lyric_emb_reduced[idx] = Ly_reduced[i]
        lyric_embeddings = lyric_emb_reduced
        print("Reduced lyric embeddings to dim", LYRIC_PCA_DIM)

    # Create dataset and dataloader (only audio for VAE training)
    dataset = MelLyricDataset(items, mel_map, lyric_embeddings=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

    # Instantiate and train Conv-VAE (audio only); skip if checkpoint exists or if --skip-train is set
    ckpt_final = os.path.join(RESULTS_DIR, "convvae_final.pt")
    model = ConvVAE(in_channels=1, latent_dim=LATENT_DIM)

    def load_latest_epoch():
        ckpts = sorted(glob.glob(os.path.join(RESULTS_DIR, "convvae_epoch*.pt")))
        return ckpts[-1] if ckpts else None

    def init_and_load(path):
        # run one forward to initialize heads, then load with strict=False to tolerate shape/layout
        sample = dataset[0]['mel']
        x = torch.from_numpy(sample).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            _ = model(x)
        state = torch.load(path, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print("Warning: unexpected keys while loading", path, unexpected)
        if missing:
            print("Warning: missing keys while loading", path, missing)
        model.to(DEVICE)

    if os.path.exists(ckpt_final):
        print("3) Conv-VAE checkpoint found; loading", ckpt_final)
        init_and_load(ckpt_final)
    elif args.skip_train:
        latest = load_latest_epoch()
        if latest:
            print("3) --skip-train set; loading latest epoch", latest)
            init_and_load(latest)
        else:
            raise SystemExit("--skip-train was set but no checkpoint found in results/")
    else:
        print(f"3) Train Conv-VAE on mel spectrograms for {epochs} epochs")
        model = train_vae(model, dataloader, epochs=epochs, lr=LR)
        torch.save(model.state_dict(), ckpt_final)

    # Extract audio latents
    print("4) Extract audio latents")
    dataset_all = MelLyricDataset(items, mel_map, lyric_embeddings=None)
    ids_order, Z_audio = extract_latents(model, dataset_all)
    print("Latent shape:", Z_audio.shape)
    np.save(os.path.join(RESULTS_DIR, "Z_audio.npy"), Z_audio)

    # Build hybrid representation: concat audio latents + reduced lyric embedding (if available)
    if lyric_embeddings is not None:
        print("5) Build hybrid (audio + lyrics) representations")
        Ly_mat = []
        for idd in ids_order:
            if idd in lyric_embeddings:
                Ly_mat.append(lyric_embeddings[idd])
            else:
                Ly_mat.append(np.zeros(LYRIC_PCA_DIM, dtype=np.float32))
        Ly_mat = np.vstack(Ly_mat)
        # Normalize both and concat
        sc1 = StandardScaler()
        sc2 = StandardScaler()
        Za = sc1.fit_transform(Z_audio)
        Zl = sc2.fit_transform(Ly_mat)
        Z_hybrid = np.concatenate([Za, Zl], axis=1)
    else:
        print("No lyrics found. Hybrid = audio latents only")
        Z_hybrid = Z_audio

    np.save(os.path.join(RESULTS_DIR, "Z_hybrid.npy"), Z_hybrid)

    # Baseline: PCA on flattened spectrogram-based features
    print("6) PCA baseline")
    ids_pca, Z_pca = pca_baseline(mel_map, items, pca_dim=LATENT_DIM)
    np.save(os.path.join(RESULTS_DIR, "Z_pca.npy"), Z_pca)

    # 7) Clustering experiments: run multiple algorithms, parameter sweep for DBSCAN and n_clusters
    print("7) Clustering experiments")
    results = []
    cluster_methods = [
        ("KMeans", {'n_clusters': N_CLUSTERS}),
        ("Agglomerative", {'n_clusters': N_CLUSTERS}),
    ]
    # DBSCAN parameter grid
    dbscan_params = [{'eps':0.5,'min_samples':5}, {'eps':1.0,'min_samples':5}, {'eps':1.5,'min_samples':5}]
    # run on hybrid and baseline
    datasets = {
        'hybrid': (Z_hybrid, ids_order),
        'pca_baseline': (Z_pca, ids_pca),
        'audio_latent': (Z_audio, ids_order)
    }
    for ds_name, (Z, ids_ds) in datasets.items():
        # KMeans & Agglomerative with varying k
        for k in [5,10,15]:
            for method, params in cluster_methods:
                mname = f"{method}_k{k}"
                metrics = run_cluster_and_eval(Z, ids_ds, labels_true_dict=labels_true,
                                               method_name=method, params=params, n_clusters=k)
                out = {
                    'dataset': ds_name,
                    'method': method,
                    'k': k,
                    'metrics': metrics
                }
                results.append(out)
                # save visualizations
                embed_and_plot(Z, metrics['labels'], ids_ds, tag=f"{ds_name}_{method}_k{k}",
                               out_prefix=os.path.join(RESULTS_DIR, ds_name))
        # DBSCAN grid
        for params in dbscan_params:
            metrics = run_cluster_and_eval(Z, ids_ds, labels_true_dict=labels_true,
                                           method_name="DBSCAN", params=params, n_clusters=None)
            out = {'dataset': ds_name, 'method':'DBSCAN', 'params': params, 'metrics':metrics}
            results.append(out)
            embed_and_plot(Z, metrics['labels'], ids_ds, tag=f"{ds_name}_DBSCAN_eps{params['eps']}",
                           out_prefix=os.path.join(RESULTS_DIR, ds_name))

    # Save metrics into a DataFrame
    rows = []
    for r in results:
        m = r['metrics']
        row = {
            'dataset': r['dataset'],
            'method': r.get('method',''),
            'k': r.get('k', np.nan),
            'params': str(r.get('params','')),
            'silhouette': m.get('silhouette', np.nan),
            'davies_bouldin': m.get('davies_bouldin', np.nan),
            'calinski_harabasz': m.get('calinski_harabasz', np.nan),
            'ARI': m.get('ARI', np.nan),
            'NMI': m.get('NMI', np.nan),
            'n_clusters_found': m.get('n_clusters_found', np.nan)
        }
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "clustering_metrics.csv"), index=False)
    print("Saved clustering_metrics.csv")

    # Save clustering labels for best method (choose top silhouette from hybrid)
    hybrid_rows = metrics_df[metrics_df['dataset']=='hybrid']
    best_row = hybrid_rows.sort_values('silhouette', ascending=False).iloc[0]
    best_method = best_row['method']
    best_k = int(best_row['k']) if not math.isnan(best_row['k']) else None
    print("Best hybrid method by silhouette:", best_method, "k=", best_k)

    # Re-run best to capture labels
    Zbest, idsbest = datasets['hybrid']
    best_metrics = run_cluster_and_eval(Zbest, idsbest, labels_true_dict=labels_true, method_name=best_method, params={}, n_clusters=best_k if best_k else N_CLUSTERS)
    label_map = dict(zip(idsbest, best_metrics['labels']))
    pd.DataFrame({'id': list(label_map.keys()), 'cluster': list(label_map.values())}).to_csv(os.path.join(RESULTS_DIR, "best_hybrid_labels.csv"), index=False)

    # Save reconstructions from VAE
    save_reconstructions(model, dataset_all, os.path.join(RESULTS_DIR, "convvae"))

    # Save final artifacts
    with open(os.path.join(RESULTS_DIR, "pipeline_artifacts.pkl"), "wb") as f:
        pickle.dump({
            'Z_audio': Z_audio,
            'Z_hybrid': Z_hybrid,
            'Z_pca': Z_pca,
            'items': items,
            'metrics_df': metrics_df
        }, f)
    print("Pipeline finished. Results in", RESULTS_DIR)

if __name__ == "__main__":
    main()