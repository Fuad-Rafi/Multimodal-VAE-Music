# notebooks/hard_pipeline.py
"""
Hard Task pipeline (joint multimodal VAE + clustering + retrieval + analysis)
- Uses existing artifacts (data/mels/, results/convvae_final.pt, results/Z_hybrid.npy, results/*.pkl) when available.
- If convvae_final.pt exists, initialize audio-encoder conv weights from it for faster convergence.
- Trains a Joint-VAE (audio encoder/decoder + lyric encoder/decoder) with shared latent space.
- Supports missing-modality training, cross-modal reconstruction, clustering, and retrieval.
- Saves: joint model, Z_joint.npy, clustering metrics, retrieval metrics, reconstructions, interpolation images.

Usage:
    python -u notebooks/hard_pipeline.py

Be sure your project layout matches:
project/
  data/
    mels/            # mel .npy files (from medium pipeline)
    lyrics.csv       # optional
    labels.csv       # optional
  results/
    convvae_final.pt (optional)
    Z_hybrid.npy (optional)
    audio_vecs.pkl (optional)
"""

import os
import glob
import math
import pickle
from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import umap
from sklearn.manifold import TSNE

# ---------- CONFIG ----------
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
MEL_DIR = DATA_DIR / "mels"
LYRICS_CSV = DATA_DIR / "lyrics.csv"
LABELS_CSV = DATA_DIR / "labels.csv"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# Model & training hyperparams (tweak as needed)
DEBUG = False   # set True to run on small subset & fewer epochs for quick debug
BATCH_SIZE = 32 if not DEBUG else 8
EPOCHS = 20 if not DEBUG else 3
LR = 1e-3
LATENT_DIM = 64
LYRIC_PCA_DIM = 64    # will compress TF-IDF / sentence embeddings to this
RECON_WEIGHT_AUDIO = 1.0
RECON_WEIGHT_LYRIC = 1.0

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# ----------------------------

# ---------- Utilities ----------
def find_mel_files(mel_dir=MEL_DIR):
    files = sorted(glob.glob(str(mel_dir / "*.npy")))
    mapping = {}
    for p in files:
        basename = Path(p).stem
        mapping[basename] = p
    return mapping

def load_labels(labels_csv=LABELS_CSV):
    if not labels_csv.exists():
        return {}
    df = pd.read_csv(labels_csv)
    out = {}
    for _, r in df.iterrows():
        out[str(r['id'])] = r['label']
    return out

def build_items_from_mels(mel_map):
    items = [{'id': k, 'mel_path': v} for k, v in sorted(mel_map.items())]
    return items

# ---------- Dataset ----------
class JointMelLyricDataset(Dataset):
    def __init__(self, items, lyric_map=None, lyric_vecs=None, use_lyrics=True, max_items=None):
        """
        items: list of {'id','mel_path'}
        lyric_map: dict id -> original lyric string (optional)
        lyric_vecs: dict id -> reduced lyric vector (optional)
        use_lyrics: whether to include lyrics (if available)
        """
        if max_items is not None:
            items = items[:max_items]
        self.items = items
        self.lyric_map = lyric_map or {}
        self.lyric_vecs = lyric_vecs or {}
        self.use_lyrics = use_lyrics

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mel = np.load(it['mel_path']).astype(np.float32)  # (n_mels, t)
        # ensure 2D shape
        if mel.ndim == 2:
            mel = mel[np.newaxis, :, :]  # (1, n_mels, t)
        # lyric vector if present
        lyric_vec = None
        has_lyric = False
        if self.use_lyrics and it['id'] in self.lyric_vecs:
            lyric_vec = self.lyric_vecs[it['id']].astype(np.float32)
            has_lyric = True
        sample = {
            'id': str(it['id']),
            'mel': mel,
            'lyric_vec': lyric_vec,
            'has_lyric': has_lyric
        }
        return sample

def collate_fn(batch):
    # batch is list of samples
    ids = [b['id'] for b in batch]
    mels = [torch.from_numpy(b['mel']) for b in batch]
    # pad variable time-dim to max width
    max_t = max([m.shape[2] for m in mels])
    padded = []
    for m in mels:
        c, h, t = m.shape
        if t < max_t:
            pad = torch.zeros((c, h, max_t - t), dtype=m.dtype)
            m_pad = torch.cat([m, pad], dim=2)
        else:
            m_pad = m
        padded.append(m_pad)
    mels_tensor = torch.stack(padded, dim=0).float()  # (B,1,H,T)
    # lyrics: build matrix with zeros where missing & mask
    lyric_vecs = [b['lyric_vec'] for b in batch]
    lyric_present = [v is not None for v in lyric_vecs]
    lyric_tensor = None
    if any(lyric_present):
        # determine lyric vector shape from first available sample
        base_vec = next(v for v in lyric_vecs if v is not None)
        zeros = torch.zeros_like(torch.from_numpy(base_vec))
        vecs = []
        for v in lyric_vecs:
            if v is not None:
                vecs.append(torch.from_numpy(v))
            else:
                vecs.append(zeros.clone())
        lyric_tensor = torch.stack(vecs, dim=0).float()
    mask = torch.tensor([1 if p else 0 for p in lyric_present], dtype=torch.float32)
    return {'ids': ids, 'mels': mels_tensor, 'lyrics': lyric_tensor, 'lyric_mask': mask}

# ---------- Models ----------
# Audio encoder: conv blocks + adaptive pooling -> vector
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class AudioEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[32,64,128,256], out_dim=LATENT_DIM*2):
        super().__init__()
        layers = []
        ch = in_channels
        for h in hidden_dims:
            layers.append(conv_block(ch, h))
            ch = h
        self.conv = nn.Sequential(*layers)
        # adaptive pooling to fixed spatial size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # ensures fixed flattened size
        flattened = ch * 4 * 4
        self.fc = nn.Linear(flattened, out_dim)

    def forward(self, x):
        # x shape: (B,1,H,T)
        h = self.conv(x)
        h = self.pool(h)
        hflat = h.view(h.size(0), -1)
        out = self.fc(hflat)
        # split mu/logvar externally
        return out, h  # returns vector and feature map for possible decoder init

class AudioDecoder(nn.Module):
    def __init__(self, out_channels=1, hidden_dims=[32,64,128,256], latent_dim=LATENT_DIM):
        super().__init__()
        # we will learn to map latent to feature-map (channels * 4 * 4)
        final_ch = hidden_dims[-1]
        self.fc = nn.Linear(latent_dim, final_ch * 4 * 4)
        # build transpose convs reversed order
        layers = []
        hidden_rev = list(reversed(hidden_dims))
        in_ch = final_ch
        for i, out_ch in enumerate(hidden_rev[1:] + [out_channels]):
            # use ConvTranspose to double spatial dims
            # last layer reduces channels to out_channels possibly with kernel=3 stride=2
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            if i < len(hidden_rev)-1:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, z, target_time=None):
        # z: (B, latent_dim)
        x = self.fc(z)
        B = z.shape[0]
        # reshape to (B, C, 4, 4)
        c = int(x.shape[1] // (4*4))
        x = x.view(B, c, 4, 4)
        x = self.deconv(x)  # (B, 1, H_out, W_out) where W_out approximates original time dim
        # our mel spectrogram original shape is (1, n_mels, t)
        # if channel dims mismatch (height), crop or pad
        return x

class LyricEncoder(nn.Module):
    def __init__(self, input_dim, hidden=256, out_dim=LATENT_DIM*2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class LyricDecoder(nn.Module):
    def __init__(self, output_dim, hidden=256, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, z):
        return self.net(z)

# Joint VAE wrapper
class JointVAE(nn.Module):
    def __init__(self, audio_enc, audio_dec, lyric_enc, lyric_dec, latent_dim=LATENT_DIM):
        super().__init__()
        self.audio_enc = audio_enc
        self.audio_dec = audio_dec
        self.lyric_enc = lyric_enc
        self.lyric_dec = lyric_dec
        self.latent_dim = latent_dim

    def encode_audio(self, x):
        # x: (B,1,H,T)
        out, feat = self.audio_enc(x)
        # out is vector of size 2*latent_dim (mu||logvar)
        mu, logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        return mu, logvar

    def encode_lyric(self, v):
        out = self.lyric_enc(v)
        mu, logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_audio(self, z, target_shape=None):
        # returns reconstructed mel (B,1,H,T')
        recon = self.audio_dec(z)
        # if target_shape given, we crop/pad width to match
        if target_shape is not None:
            B, C, H, W = recon.shape
            _, _, Ht, Wt = target_shape
            # crop/pad height and width as needed
            # crop height
            if H > Ht:
                hstart = (H - Ht)//2
                recon = recon[:, :, hstart:hstart+Ht, :]
            elif H < Ht:
                padh = Ht - H
                recon = nn.functional.pad(recon, (0,0,0, padh))
            # width
            if W > Wt:
                wstart = (W - Wt)//2
                recon = recon[:, :, :, wstart:wstart+Wt]
            elif W < Wt:
                padw = Wt - W
                recon = nn.functional.pad(recon, (0, padw, 0,0))
        return recon

    def decode_lyric(self, z):
        return self.lyric_dec(z)

    def forward(self, mels, lyric_vecs=None, lyric_mask=None):
        """
        mels: (B,1,H,T)
        lyric_vecs: (B, L) or None
        lyric_mask: (B,) float mask 1.0 if lyric present else 0.0
        returns recon_audio, recon_lyric, mu, logvar (joint)
        """
        batch_size = mels.size(0)
        # audio encode
        mu_a, logvar_a = self.encode_audio(mels)
        # lyric encode if present
        has_lyrics = lyric_vecs is not None and lyric_mask is not None
        if has_lyrics:
            mu_l, logvar_l = self.encode_lyric(lyric_vecs)
            mask = lyric_mask.view(-1, 1)
            mask_bool = mask > 0.5
            # use audio-only where mask=0, average audio+lyric where mask=1
            mu = torch.where(mask_bool, (mu_a + mu_l) / 2.0, mu_a)
            logvar = torch.where(mask_bool, (logvar_a + logvar_l) / 2.0, logvar_a)
        else:
            if lyric_mask is None:
                lyric_mask = torch.zeros((batch_size,), device=mels.device)
            mu = mu_a
            logvar = logvar_a

        z = self.reparam(mu, logvar)
        # decode both
        recon_audio = self.decode_audio(z, target_shape=mels.shape)
        recon_lyric = None
        if lyric_vecs is not None:
            recon_lyric = self.decode_lyric(z)
        return recon_audio, recon_lyric, mu, logvar

# ---------- Loss helpers ----------
def kl_divergence(mu, logvar):
    # per-batch sum
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def joint_loss(recon_audio, audio_target, recon_lyric, lyric_target, mu, logvar, mask):
    # audio MSE (sum)
    recon_audio_loss = nn.functional.mse_loss(recon_audio, audio_target, reduction='sum')
    recon_lyric_loss = 0.0
    if (recon_lyric is not None) and (lyric_target is not None):
        # only compute for samples with lyric (mask)
        # mask shape (B,)
        mask = mask.view(-1,1)
        # multiply per-sample
        diff = (recon_lyric - lyric_target) * mask
        recon_lyric_loss = torch.sum(diff * diff)
    kld = kl_divergence(mu, logvar)
    loss = RECON_WEIGHT_AUDIO * recon_audio_loss + RECON_WEIGHT_LYRIC * recon_lyric_loss + kld
    return loss, recon_audio_loss, recon_lyric_loss, kld

# ---------- Helper: initialize audio encoder from convvae if available ----------
def try_load_convvae_to_audio_encoder(audio_encoder, convvae_path):
    """
    If a convVAE checkpoint exists, try to load matching conv layers' weights to audio_encoder.conv
    This is best-effort: we map parameters by layer order.
    """
    if not convvae_path.exists():
        print("No convvae checkpoint found at", convvae_path)
        return False
    try:
        ckpt = torch.load(str(convvae_path), map_location='cpu')
    except Exception as e:
        print("Failed loading convvae checkpoint:", e)
        return False
    # the convVAE implementation used sequential conv layers; we try to map conv weight shapes
    # this is fragile but improves initialization if shapes match
    conv_state = audio_encoder.conv.state_dict()
    # find candidate parameters in ckpt with 'enc' or 'conv' names
    matched = 0
    for k_src, v_src in ckpt.items():
        # This checkpoint likely contains 'enc.X.weight' naming; try to match by shapes only
        if 'enc' in k_src or 'conv' in k_src:
            # attempt to find a target param with same shape
            for k_tgt, v_tgt in conv_state.items():
                if v_src.shape == v_tgt.shape:
                    conv_state[k_tgt] = v_src.clone()
                    matched += 1
                    break
        if matched >= len(conv_state):
            break
    audio_encoder.conv.load_state_dict(conv_state)
    print(f"Initialized audio encoder conv layers from {convvae_path} (matched {matched} params)")
    return True

# ---------- Training & utilities ----------
def train_joint_vae(model, train_loader, epochs=EPOCHS, lr=LR, device=DEVICE, save_every=10):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        total_audio = 0.0
        total_lyric = 0.0
        total_kld = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            ids = batch['ids']
            mels = batch['mels'].to(device)  # (B,1,H,T)
            lyrics = batch['lyrics'].to(device) if batch['lyrics'] is not None else None
            mask = batch['lyric_mask'].to(device)
            opt.zero_grad()
            recon_a, recon_l, mu, logvar = model(mels, lyric_vecs=lyrics, lyric_mask=mask)
            loss, ra, rl, kld = joint_loss(recon_a, mels, recon_l, lyrics, mu, logvar, mask)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_audio += ra.item() if isinstance(ra, torch.Tensor) else ra
            total_lyric += rl if isinstance(rl, float) else (rl.item() if isinstance(rl, torch.Tensor) else rl)
            total_kld += kld.item() if isinstance(kld, torch.Tensor) else kld
        print(f"Epoch {epoch} total_loss={total_loss/len(train_loader.dataset):.4f} audio={total_audio:.1f} lyric={total_lyric:.1f} kld={total_kld:.1f}")
        if epoch % save_every == 0 or epoch == epochs:
            torch.save(model.state_dict(), str(RESULTS_DIR / f"jointvae_epoch{epoch}.pt"))
    torch.save(model.state_dict(), str(RESULTS_DIR / "jointvae_final.pt"))
    return model

def extract_joint_latents(model, dataset, device=DEVICE):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    ids_all = []
    mus = []
    with torch.no_grad():
        for batch in loader:
            mels = batch['mels'].to(device)
            lyrics = batch['lyrics'].to(device) if batch['lyrics'] is not None else None
            mask = batch['lyric_mask'].to(device)
            # obtain mu from components; we re-use encode_audio/encode_lyric logic
            mu_a, _ = model.encode_audio(mels)
            if lyrics is not None:
                mu_l, _ = model.encode_lyric(lyrics)
                mu_joint = (mu_a + mu_l) / 2.0
            else:
                mu_joint = mu_a
            mus.append(mu_joint.cpu().numpy())
            ids_all.extend(batch['ids'])
    Z = np.vstack(mus)
    return ids_all, Z

# ---------- Clustering & metrics ----------
def cluster_and_eval(Z, ids, labels_true, dataset_name="joint", methods=[("KMeans",10)], results_path=RESULTS_DIR):
    rows = []
    for method, param in methods:
        labels = None
        params_str = ""
        k_val = np.nan
        if method == "KMeans":
            k_val = int(param)
            km = KMeans(n_clusters=k_val, random_state=RANDOM_SEED, n_init=10).fit(Z)
            labels = km.labels_
        elif method == "Agglomerative":
            k_val = int(param)
            lab = AgglomerativeClustering(n_clusters=k_val).fit_predict(Z)
            labels = lab
        elif method == "DBSCAN":
            eps = param.get('eps', 0.5) if isinstance(param, dict) else 0.5
            min_samples = param.get('min_samples', 5) if isinstance(param, dict) else 5
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(Z)
            params_str = f"eps={eps},min_samples={min_samples}"
        else:
            raise NotImplementedError
        # metrics
        unique = np.unique(labels)
        sil = silhouette_score(Z, labels) if len(unique)>1 else float('nan')
        dbi = davies_bouldin_score(Z, labels) if len(unique)>1 else float('nan')
        ch = calinski_harabasz_score(Z, labels) if len(unique)>1 else float('nan')
        ari = float('nan')
        nmi = float('nan')
        if labels_true:
            true = [labels_true.get(i, -1) for i in ids]
            mask = np.array([t != -1 for t in true])
            if mask.sum() > 1:
                ari = adjusted_rand_score(np.array(true)[mask], np.array(labels)[mask])
                nmi = normalized_mutual_info_score(np.array(true)[mask], np.array(labels)[mask])
        rows.append({
            'dataset': dataset_name,
            'method': method,
            'k': k_val,
            'params': params_str,
            'silhouette': sil,
            'davies_bouldin': dbi,
            'calinski_harabasz': ch,
            'ARI': ari,
            'NMI': nmi,
            'n_samples': Z.shape[0]
        })
        # save cluster labels file
        suffix = f"k{k_val}" if not math.isnan(k_val) else (params_str.replace(',','_') if params_str else "dbscan")
        out_df = pd.DataFrame({'id': ids, 'cluster': labels})
        out_df.to_csv(results_path / f"{dataset_name}_{method}_{suffix}_labels.csv", index=False)
    return pd.DataFrame(rows)

# ---------- Retrieval (lyric->audio) ----------
def cross_modal_retrieval(model, items_dataset, lyric_vecs_dict, labels_true, ids_order, Z_audio_latent=None, topk=(1,5,10)):
    """
    For each lyric sample (with lyric vector), encode lyric to joint mu, then find nearest audio latent mus (Z_audio_latent)
    If Z_audio_latent not given, we compute audio mus from model on all items.
    Returns recall@k
    """
    # compute audio mus if not provided
    if Z_audio_latent is None:
        _, Z_audio_latent = extract_joint_latents(model, items_dataset)
    # build neighbor index
    nbr = NearestNeighbors(n_neighbors=max(topk), algorithm='auto').fit(Z_audio_latent)
    # build lyric list with ids
    lyric_ids = [iid for iid, v in lyric_vecs_dict.items()]
    lyric_vecs = np.vstack([lyric_vecs_dict[i] for i in lyric_ids])
    # encode lyrics to mu (batch)
    model.eval()
    mus = []
    with torch.no_grad():
        for i in range(0, lyric_vecs.shape[0], BATCH_SIZE):
            batch = torch.from_numpy(lyric_vecs[i:i+BATCH_SIZE]).float().to(DEVICE)
            mu_l, _ = model.encode_lyric(batch)
            mus.append(mu_l.cpu().numpy())
    lyric_mus = np.vstack(mus)
    # query neighbors
    neighs = nbr.kneighbors(lyric_mus, return_distance=False)
    # compute recall@k if labels_true available
    recalls = {k: 0 for k in topk}
    total = len(lyric_ids)
    if labels_true:
        for i, lid in enumerate(lyric_ids):
            true_label = labels_true.get(lid, None)
            if true_label is None:
                total -= 1
                continue
            retrieved_ids = [ids_order[idx] for idx in neighs[i]]
            for k in topk:
                if any(labels_true.get(rid, None) == true_label for rid in retrieved_ids[:k]):
                    recalls[k] += 1
    # compute percentages
    recall_at_k = {k: (recalls[k] / total * 100.0) if total > 0 else float('nan') for k in topk}
    return recall_at_k, neighs, lyric_ids

# ---------- Visualizations ----------
def save_reconstructions(model, dataset, out_path, n_examples=8):
    model.eval()
    os.makedirs(out_path, exist_ok=True)
    # pick n randomsamples
    idxs = np.random.choice(len(dataset), size=min(n_examples, len(dataset)), replace=False)
    fig, axes = plt.subplots(2, len(idxs), figsize=(3*len(idxs), 6))
    for i, idx in enumerate(idxs):
        s = dataset[idx]
        mel = torch.from_numpy(s['mel']).unsqueeze(0).to(DEVICE).float()  # (1,1,H,T)
        lyric_vec = torch.from_numpy(s['lyric_vec']).unsqueeze(0).to(DEVICE).float() if s['lyric_vec'] is not None else None
        mask = torch.tensor([1.0], device=DEVICE) if s['lyric_vec'] is not None else torch.tensor([0.0], device=DEVICE)
        with torch.no_grad():
            recon_a, recon_l, mu, logvar = model(mel, lyric_vecs=lyric_vec, lyric_mask=mask)
        orig = mel.cpu().numpy()[0,0]
        recon = recon_a.cpu().numpy()[0,0]
        ax1 = axes[0,i]
        ax2 = axes[1,i]
        librosa.display.specshow(orig, y_axis='mel', x_axis='time', hop_length=512, ax=ax1)
        ax1.set_title("Original")
        librosa.display.specshow(recon, y_axis='mel', x_axis='time', hop_length=512, ax=ax2)
        ax2.set_title("Recon")
    plt.tight_layout()
    plt.savefig(out_path / "joint_reconstructions.png", dpi=150)
    plt.close()

def save_interpolations(model, dataset, out_path, n_pairs=4, steps=6):
    model.eval()
    os.makedirs(out_path, exist_ok=True)
    # randomly sample pairs
    idxs = np.random.choice(len(dataset), size=min(2*n_pairs, len(dataset)), replace=False)
    pairs = [(idxs[i], idxs[i+1]) for i in range(0, len(idxs)-1, 2)][:n_pairs]
    fig, axes = plt.subplots(n_pairs, steps+2, figsize=(3*(steps+2), 3*n_pairs))
    for r, (i1, i2) in enumerate(pairs):
        s1 = dataset[i1]
        s2 = dataset[i2]
        m1 = torch.from_numpy(s1['mel']).unsqueeze(0).to(DEVICE).float()
        m2 = torch.from_numpy(s2['mel']).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad():
            mu1, _ = model.encode_audio(m1)
            mu2, _ = model.encode_audio(m2)
        for k, alpha in enumerate(np.linspace(0, 1, steps+2)):
            z = (1-alpha) * mu1 + alpha * mu2
            with torch.no_grad():
                recon = model.decode_audio(z, target_shape=m1.shape).cpu().numpy()[0,0]
            ax = axes[r,k]
            librosa.display.specshow(recon, y_axis='mel', x_axis='time', hop_length=512, ax=ax)
            ax.set_title(f"alpha={alpha:.2f}")
    plt.tight_layout()
    plt.savefig(out_path / "interpolations.png", dpi=150)
    plt.close()

def plot_latent_embeddings(Z, labels, out_path, tag="joint"):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
    Z2 = reducer.fit_transform(Z)
    plt.figure(figsize=(8,6))
    plt.scatter(Z2[:,0], Z2[:,1], c=labels, s=8, cmap='tab10')
    plt.title(f"UMAP: {tag}")
    plt.tight_layout()
    plt.savefig(out_path / f"{tag}_umap.png", dpi=150)
    plt.close()

    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, init='pca', learning_rate='auto')
    Zt = tsne.fit_transform(Z)
    plt.figure(figsize=(8,6))
    plt.scatter(Zt[:,0], Zt[:,1], c=labels, s=8, cmap='tab10')
    plt.title(f"t-SNE: {tag}")
    plt.tight_layout()
    plt.savefig(out_path / f"{tag}_tsne.png", dpi=150)
    plt.close()

# ---------- Main pipeline ----------
def main():
    print("=== Hard Task pipeline started ===")
    mel_map = find_mel_files()
    print(f"Found {len(mel_map)} mel files")
    if len(mel_map) == 0:
        raise RuntimeError("No mel .npy files found in data/mels/. Run the medium pipeline to create them first.")

    items = build_items_from_mels(mel_map)
    if DEBUG:
        items = items[:50]

    # load existing artifacts if present
    convvae_ckpt = RESULTS_DIR / "convvae_final.pt"
    z_hybrid_path = RESULTS_DIR / "Z_hybrid.npy"
    audio_vecs_path = RESULTS_DIR / "audio_vecs.pkl"

    labels_true = load_labels()

    # build lyric embeddings (TF-IDF -> PCA) if lyrics.csv exists
    lyric_vecs = {}
    if LYRICS_CSV.exists():
        print("Building TF-IDF + PCA lyric embeddings (will reuse if previously saved)")
        lyrics_df = pd.read_csv(LYRICS_CSV)
        # align IDs to items
        lyrics_df['id'] = lyrics_df['id'].astype(str)
        # build TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        txts = lyrics_df['lyrics'].fillna("").astype(str).tolist()
        ids = lyrics_df['id'].tolist()
        tfv = TfidfVectorizer(max_features=4096)
        X = tfv.fit_transform(txts).toarray().astype(np.float32)
        # PCA reduce
        pca = PCA(n_components=LYRIC_PCA_DIM, random_state=RANDOM_SEED)
        Xred = pca.fit_transform(X)
        lyric_vecs = {str(ids[i]): Xred[i] for i in range(len(ids))}
        print("Lyric embeddings built for", len(lyric_vecs))
    else:
        print("No lyrics.csv found — training will proceed in audio-only mode (joint model will still be audio-first).")

    # create dataset & dataloader
    dataset = JointMelLyricDataset(items, lyric_vecs=lyric_vecs, use_lyrics=bool(lyric_vecs), max_items=None)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # build models
    print("Building models...")
    audio_enc = AudioEncoder(in_channels=1)
    audio_dec = AudioDecoder(out_channels=1, latent_dim=LATENT_DIM)
    lyric_enc = LyricEncoder(input_dim=LYRIC_PCA_DIM, out_dim=LATENT_DIM*2) if lyric_vecs else None
    lyric_dec = LyricDecoder(output_dim=LYRIC_PCA_DIM, latent_dim=LATENT_DIM) if lyric_vecs else None

    joint = JointVAE(audio_enc, audio_dec, lyric_enc, lyric_dec, latent_dim=LATENT_DIM)

    # try to initialize audio encoder from convVAE checkpoint if available
    if convvae_ckpt.exists():
        print("Attempting to initialize audio encoder from", convvae_ckpt)
        try_load_convvae_to_audio_encoder(audio_enc, convvae_ckpt)

    # train joint VAE
    print("Training joint VAE (this may take time)...")
    joint = train_joint_vae(joint, train_loader, epochs=EPOCHS, lr=LR, device=DEVICE, save_every=max(1, EPOCHS//5))

    # extract joint latents (mu)
    print("Extracting joint latents...")
    ids_order, Z_joint = extract_joint_latents(joint, dataset)
    print("Z_joint shape:", Z_joint.shape)
    np.save(RESULTS_DIR / "Z_joint.npy", Z_joint)

    # Save mapping id->latent index
    pd.DataFrame({'id': ids_order}).to_csv(RESULTS_DIR / "Z_joint_ids.csv", index=False)

    # clustering experiments using joint latents (reuse functions)
    print("Running clustering experiments on joint latents...")
    cluster_methods = [
        ("KMeans", 10), ("KMeans", 5), ("Agglomerative", 10),
        ("DBSCAN", {'eps': 0.5, 'min_samples': 5}),
        ("DBSCAN", {'eps': 1.0, 'min_samples': 5}),
        ("DBSCAN", {'eps': 1.5, 'min_samples': 5}),
    ]
    metrics_df = cluster_and_eval(Z_joint, ids_order, labels_true, dataset_name="joint", methods=cluster_methods, results_path=RESULTS_DIR)
    metrics_df.to_csv(RESULTS_DIR / "hard_joint_clustering_metrics.csv", index=False)
    print("Saved clustering metrics to results/hard_joint_clustering_metrics.csv")

    # choose best by silhouette for visualization
    metrics_sorted = metrics_df.sort_values('silhouette', ascending=False)
    labels_for_plot = None
    if not metrics_sorted.empty:
        best_row = metrics_sorted.iloc[0]
        method = best_row['method']
        if method == "KMeans" and not math.isnan(best_row['k']):
            km = KMeans(n_clusters=int(best_row['k']), random_state=RANDOM_SEED, n_init=10).fit(Z_joint)
            labels_for_plot = km.labels_
        elif method == "Agglomerative" and not math.isnan(best_row['k']):
            labels_for_plot = AgglomerativeClustering(n_clusters=int(best_row['k'])).fit_predict(Z_joint)
        elif method == "DBSCAN":
            params = {}
            if isinstance(best_row['params'], str) and best_row['params']:
                for tok in best_row['params'].split(','):
                    if '=' in tok:
                        k, v = tok.split('=')
                        params[k.strip()] = float(v)
            eps = params.get('eps', 0.5)
            min_samples = int(params.get('min_samples', 5))
            labels_for_plot = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z_joint)
        if labels_for_plot is not None:
            plot_latent_embeddings(Z_joint, labels_for_plot, RESULTS_DIR, tag=f"joint_{method}")

    # Cross-modal retrieval (if lyrics exist)
    retrieval_metrics = {}
    if lyric_vecs:
        print("Running cross-modal retrieval (lyrics -> audio)...")
        recall_at_k, neighs, lyric_ids = cross_modal_retrieval(joint, dataset, lyric_vecs, labels_true, ids_order, Z_audio_latent=Z_joint, topk=(1,5,10))
        retrieval_metrics = {'recall_at_k': recall_at_k}
        print("Retrieval recall@k:", recall_at_k)
        # save retrieval neighbor indices for inspection
        with open(RESULTS_DIR / "retrieval_neighs.pkl", "wb") as f:
            pickle.dump({'lyric_ids': lyric_ids, 'neighs': neighs}, f)
        pd.DataFrame([{'k': k, 'recall': v} for k, v in recall_at_k.items()]).to_csv(RESULTS_DIR / "retrieval_metrics.csv", index=False)
    else:
        print("No lyrics - skipping cross-modal retrieval.")

    # Save reconstructions and interpolations
    print("Saving reconstructions and interpolations...")
    save_reconstructions(joint, dataset, RESULTS_DIR, n_examples=8)
    save_interpolations(joint, dataset, RESULTS_DIR, n_pairs=4, steps=6)

    # Save final model and artifacts
    torch.save(joint.state_dict(), RESULTS_DIR / "jointvae_final.pt")
    with open(RESULTS_DIR / "hard_pipeline_artifacts.pkl", "wb") as f:
        pickle.dump({
            'ids_order': ids_order,
            'Z_joint': Z_joint,
            'clustering_metrics': metrics_df.to_dict(orient='list'),
            'retrieval_metrics': retrieval_metrics
        }, f)
    print("Saved joint model and artifacts to results/")

    print("=== Hard Task pipeline finished ===")
    print("Artifacts you may want to include in the report:")
    print(" - results/jointvae_final.pt")
    print(" - results/Z_joint.npy and results/Z_joint_ids.csv")
    print(" - results/hard_joint_clustering_metrics.csv")
    if lyric_vecs:
        print(" - results/retrieval_metrics.csv and results/retrieval_neighs.pkl")
    print(" - results/joint_reconstructions.png")
    print(" - results/interpolations.png")

if __name__ == "__main__":
    main()
