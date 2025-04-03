import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import time

from EgoExoEMS.EgoExoEMS import CLIP_EgoExo_Keystep_Dataset, clip_collate_fn


# 1️⃣ **Temporal Encoder Using GRU**
class TemporalEgoExoEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128, num_layers=2, bidirectional=True):
        super(TemporalEgoExoEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x, lengths):
        lengths = lengths.cpu().to(torch.int64)

        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        last_outputs = out[torch.arange(out.shape[0]), lengths - 1, :]
        return self.fc(last_outputs)


# 2️⃣ **Triplet Loss Function**
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1)

        loss = torch.clamp(self.margin + neg_sim - pos_sim, min=0.0)
        return loss.mean()


# 3️⃣ **Training Function (DDP)**
def train(rank, world_size, args):
    """ Runs the training on a given GPU rank """
    print(f"Starting DDP training on rank {rank}")
    
    # Set up distributed processing
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Load dataset and use DistributedSampler
    dataset = CLIP_EgoExo_Keystep_Dataset(args.annotation_file, args.data_base_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // world_size,  # Reduce batch size per GPU
        collate_fn=clip_collate_fn,
        shuffle=False,  # DistributedSampler already shuffles
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Create model and move to GPU
    model = TemporalEgoExoEncoder(input_dim=512).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Loss function & Optimizer
    loss_fn = TripletLoss(margin=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # Ensure proper shuffling across epochs
        total_loss = 0.0

        for batch in dataloader:
            ego_clip = batch["ego_clip"].to(device, non_blocking=True)
            exo_clip = batch["exo_clip"].to(device, non_blocking=True)
            neg_exo_clip = batch["neg_exo_clip"].to(device, non_blocking=True)

            ego_lengths = torch.tensor([x.shape[0] for x in batch["ego_clip"]], dtype=torch.int64, device="cpu")
            exo_lengths = torch.tensor([x.shape[0] for x in batch["exo_clip"]], dtype=torch.int64, device="cpu")
            neg_exo_lengths = torch.tensor([x.shape[0] for x in batch["neg_exo_clip"]], dtype=torch.int64, device="cpu")

            optimizer.zero_grad()
            ego_emb = model(ego_clip, ego_lengths)
            exo_emb = model(exo_clip, exo_lengths)
            neg_exo_emb = model(neg_exo_clip, neg_exo_lengths)

            loss = loss_fn(ego_emb, exo_emb, neg_exo_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Rank {rank}, Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

        if rank == 0:
            torch.save(model.module.state_dict(), f"./chkpoints/{args.job_id}_cross_retriever_{epoch+1}.pt")

    dist.destroy_process_group()


# 4️⃣ **Main Function for DDP**
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default="0")
    parser.add_argument("--annotation_file", type=str, default="../../Annotations/splits/trials/train_split_classification.json")
    parser.add_argument("--data_base_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=128)  # Global batch size
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()

    # Number of GPUs
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Less than 2 GPUs detected! Use DataParallel instead.")
        exit(1)

    # Get rank from environment variables (torchrun passes LOCAL_RANK)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Run training
    train(local_rank, world_size, args)
    print("Training completed.")