import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random

# Import your dataset class
from EgoExoEMS import CLIP_EgoExo_Keystep_Dataset  # Make sure dataset.py contains your dataset class

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1️⃣ Model Definition: Simple MLP-based Feature Encoder
class EgoExoEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super(EgoExoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# 2️⃣ Contrastive Loss Function (Triplet Loss)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


# 3️⃣ Training Function
def train(model, dataloader, optimizer, loss_fn, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            ego_clip = batch["ego_clip"].to(device)  # (batch_size, num_keystep_frames, feature_dim)
            exo_clip = batch["exo_clip"].to(device)
            neg_exo_clip = batch["neg_exo_clip"].to(device)

            # Average across frames to get per-keystep representations
            ego_feat = ego_clip.mean(dim=1)  # (batch, feature_dim)
            exo_feat = exo_clip.mean(dim=1)
            neg_exo_feat = neg_exo_clip.mean(dim=1)

            # Forward pass through the encoder
            ego_emb = model(ego_feat)
            exo_emb = model(exo_feat)
            neg_exo_emb = model(neg_exo_feat)

            # Compute loss
            loss = loss_fn(ego_emb, exo_emb, neg_exo_emb)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training Complete!")


# 4️⃣ Retrieval Function: Given an Ego Query, Retrieve the Closest Exo Clip
def retrieve_exo_clip(model, ego_query, exo_features, exo_clip_ids, top_k=5):
    """
    ego_query: (feature_dim,) tensor -> Input ego clip feature vector
    exo_features: (num_exo_clips, feature_dim) tensor -> All available exo features
    exo_clip_ids: List -> Corresponding exo clip IDs
    top_k: Number of top results to return
    """
    model.eval()
    with torch.no_grad():
        ego_query = model(ego_query.unsqueeze(0))  # Encode query (1, feature_dim)
        exo_features = model(exo_features)  # Encode all exo clips (num_exo_clips, feature_dim)

        # Compute cosine similarity
        similarities = F.cosine_similarity(ego_query, exo_features)  # (num_exo_clips,)

        # Retrieve top-k exo clips
        top_k_indices = similarities.topk(k=top_k, largest=True).indices
        retrieved_exo_clips = [exo_clip_ids[i] for i in top_k_indices]

    return retrieved_exo_clips


# 5️⃣ Load Dataset and Train the Model
if __name__ == "__main__":
    # Dataset Paths
    annotation_file = "annotations.json"
    annotation_file = "../../Annotations/splits/keysteps/train_split.json"  # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)

    data_base_path = ""

    # Create dataset & dataloader
    dataset = CLIP_EgoExo_Keystep_Dataset(annotation_file, data_base_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model, Loss, Optimizer
    input_dim = 512  # CLIP feature dimension
    model = EgoExoEncoder(input_dim=input_dim).to(device)
    loss_fn = ContrastiveLoss(margin=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the Model
    train(model, dataloader, optimizer, loss_fn, num_epochs=20)

    # Example Retrieval:
    # Load some ego features and all exo features for retrieval
    ego_query_feat = torch.randn(input_dim).to(device)  # Simulated ego feature
    exo_features = torch.randn(len(dataset), input_dim).to(device)  # Simulated all exo features
    exo_clip_ids = [i for i in range(len(dataset))]  # Simulated clip IDs

    retrieved_exo_clips = retrieve_exo_clip(model, ego_query_feat, exo_features, exo_clip_ids, top_k=5)
    print("Top Retrieved Exo Clips:", retrieved_exo_clips)
