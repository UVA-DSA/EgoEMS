import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

import time
# Import the updated dataset class
from EgoExoEMS.EgoExoEMS import CLIP_EgoExo_Keystep_Dataset, clip_collate_fn, CLIP_EgoExo_Keystep_LIMITED_Dataset

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# 1️⃣ **Temporal Encoder Using GRU**
class TemporalEgoExoEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128, num_layers=2, bidirectional=True):
        super(TemporalEgoExoEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU for temporal modeling
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

        # Final projection layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x, lengths):
        """
        x: (batch, seq_len, feature_dim)
        lengths: (batch,) -> Actual lengths of sequences before padding
        """
        # Ensure lengths is a CPU int64 tensor
        lengths = lengths.cpu().to(torch.int64)

        # Pack sequence
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward through GRU
        packed_out, _ = self.gru(packed_x)  

        # Unpack sequence
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Get last valid hidden state for each sequence
        last_outputs = out[torch.arange(out.shape[0]), lengths - 1, :]

        return self.fc(last_outputs)  # Final feature representation


# 2️⃣ **Triplet Loss Function**
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        anchor: (batch, feature_dim)
        positive: (batch, feature_dim)
        negative: (batch, feature_dim)
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        # Compute cosine similarity
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)  # (batch,)
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1)  # (batch,)

        # Compute triplet loss
        loss = torch.clamp(self.margin + neg_sim - pos_sim, min=0.0)
        return loss.mean()


# 3️⃣ **Training Function**
def train(model, dataloader, optimizer, loss_fn, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            ego_clip = batch["ego_clip"].to(device)  # (batch_size, seq_len, feature_dim)
            exo_clip = batch["exo_clip"].to(device)
            neg_exo_clip = batch["neg_exo_clip"].to(device)

            start_t = time.time()
            ego_lengths = torch.tensor([x.shape[0] for x in batch["ego_clip"]], dtype=torch.int64, device='cpu')
            exo_lengths = torch.tensor([x.shape[0] for x in batch["exo_clip"]], dtype=torch.int64, device='cpu')
            neg_exo_lengths = torch.tensor([x.shape[0] for x in batch["neg_exo_clip"]], dtype=torch.int64, device='cpu')

            # Forward pass through GRU-based encoder
            ego_emb = model(ego_clip, ego_lengths)
            exo_emb = model(exo_clip, exo_lengths)
            neg_exo_emb = model(neg_exo_clip, neg_exo_lengths)

            # Compute triplet loss
            loss = loss_fn(ego_emb, exo_emb, neg_exo_emb)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_t = time.time()
            
            print(f"Batch Time: {end_t - start_t:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save model
        torch.save(model.state_dict(), f"./chkpoints/{job_id}_cross_retriever_{epoch+1}.pt")

    print("Training Complete!")


# 4️⃣ **Retrieval Function**
def retrieve_exo_clip(model, ego_query, exo_features, exo_clip_ids, top_k=5):
    """
    ego_query: (seq_len, feature_dim) tensor -> Input ego clip feature sequence
    exo_features: (num_exo_clips, seq_len, feature_dim) tensor -> All available exo clip features
    exo_clip_ids: List -> Corresponding exo clip IDs
    top_k: Number of top results to return
    """
    model.eval()
    with torch.no_grad():
        lengths = torch.tensor([ego_query.shape[0]]).to(device)  # Single clip length
        ego_query = model(ego_query.unsqueeze(0), lengths)  # Encode query (1, feature_dim)

        lengths_exo = torch.tensor([x.shape[0] for x in exo_features]).to(device)
        exo_features = model(exo_features, lengths_exo)  # Encode all exo clips (num_exo_clips, feature_dim)

        # Compute cosine similarity
        similarities = F.cosine_similarity(ego_query, exo_features)  # (num_exo_clips,)

        # Retrieve top-k exo clips
        top_k_indices = similarities.topk(k=top_k, largest=True).indices
        retrieved_exo_clips = [exo_clip_ids[i] for i in top_k_indices]

    return retrieved_exo_clips


# 5️⃣ **Load Dataset and Train the Model**
if __name__ == "__main__":

    # get arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default="0")

    args = parser.parse_args()
    job_id = args.job_id

    # Dataset Paths
    annotation_file = "../../Annotations/splits/trials/train_split_classification.json"  # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)
    data_base_path = ""

    # Create dataset & dataloader
    dataset = CLIP_EgoExo_Keystep_Dataset(annotation_file, data_base_path)
    # dataset = CLIP_EgoExo_Keystep_LIMITED_Dataset(annotation_file=annotation_file, fps= 29.97, data_base_path="",max_neg_samples=300,max_pos_samples=300)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=clip_collate_fn, shuffle=True, num_workers=8, pin_memory=True)

    # print stats of the dataset

    print("Dataset Stats:")
    print("Number of triplet pairs:", len(dataset))
    print("Number of batches:", len(dataloader))   

    # Model, Loss, Optimizer
    input_dim = 512  # CLIP feature dimension
    model = TemporalEgoExoEncoder(input_dim=input_dim).to(device)
    loss_fn = TripletLoss(margin=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the Model
    train(model, dataloader, optimizer, loss_fn, num_epochs=20)

    # Example Retrieval:
    # ego_query_feat = torch.randn(30, input_dim).to(device)  # Simulated ego feature sequence (30 frames)
    # exo_features = torch.randn(len(dataset), 30, input_dim).to(device)  # Simulated exo feature sequences
    # exo_clip_ids = [i for i in range(len(dataset))]  # Simulated clip IDs

    # retrieved_exo_clips = retrieve_exo_clip(model, ego_query_feat, exo_features, exo_clip_ids, top_k=5)
    # print("Top Retrieved Exo Clips:", retrieved_exo_clips)

