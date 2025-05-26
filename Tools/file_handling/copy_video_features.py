import os
import glob

# ─── CONFIG: set to None to disable that branch ──────────────────────────────
ego_file_list_path = "/home/cjh9fw/Desktop/2024/repos/video_features/videos_to_extract/opvrs_ego_file_paths_p2.txt"   # ← set to None if you don’t want ego
exo_file_list_path = None   # ← set to None if you don’t want exo

video_feature_directory = "/home/cjh9fw/Desktop/2024/repos/video_features/extracted_features/opvrs/opvrs_ego_file_paths_p2/resnet/resnet50"


# ─── LOAD FILE‑PATH MAPS ─────────────────────────────────────────────────────
def load_mapping(list_path):
    """
    Reads a file of mp4 paths (one per line) and returns
    a dict mapping basename (sans “.mp4”) → full path.
    """
    mapping = {}
    if list_path is None:
        return mapping
    with open(list_path, "r") as f:
        for line in f:
            path = line.strip()
            key = os.path.basename(path).rsplit(".mp4", 1)[0]
            mapping[key] = path
    return mapping


# helper to do the actual copying
def handle_branch(mapping, tag):

    if video_name not in mapping:
        return
    vid_path = mapping[video_name]
    # go up two levels from the GoPro file to get subject/trial folder
    base_dir = os.path.dirname(os.path.dirname(vid_path))
    target_dir = os.path.join(base_dir, f"resnet_{tag}")
    os.makedirs(target_dir, exist_ok=True)

    # extract subject & trial for naming
    parts  = base_dir.split(os.sep)
    subject, scenario, trial = parts[-3], parts[-2], parts[-1]

    dst_name = f"{subject}_{scenario}_{trial}_{tag}_resnet.npy"
    dst_path = os.path.join(target_dir, dst_name)

    print(f"dst_path: {dst_path}")
    print(f"[{tag.upper()}] copying {basename} → {dst_path}")
    os.system(f'cp "{feat_path}" "{dst_path}"')
    


if __name__ == "__main__":

    print(f"Copying video features from {video_feature_directory}...")

    # load file-path maps
    ego_map = load_mapping(ego_file_list_path)
    exo_map = load_mapping(exo_file_list_path)


    # process each feature file
    for feat_path in glob.glob(f"{video_feature_directory}/*_resnet.npy"):
        basename    = os.path.basename(feat_path)
        video_name  = basename.split("_resnet.npy")[0]

        print("-" * 20)
        print(f"Processing {basename}...")
        # run ego branch only if ego_map was loaded
        if len(ego_map) > 0:
            handle_branch(ego_map, "ego")

        # run exo branch only if exo_map was loaded
        if len(exo_map) > 0:
            handle_branch(exo_map, "exo")

        print("-" * 20)
        