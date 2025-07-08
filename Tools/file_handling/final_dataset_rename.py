#!/usr/bin/env python3
import os
import argparse

def rename_files(root_dir):
    """
    Walk through the dataset directory and rename files so that every file
    within a modality is prefixed with {subject}_{scenario}_{trial}_.
    """
    for subject in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        # subject directories (e.g., cars_1, cars_2, ...)
        for scenario in os.listdir(subject_path):
            scenario_path = os.path.join(subject_path, scenario)
            if not os.path.isdir(scenario_path):
                continue
            # scenarios (e.g., stroke, chest_pain, cardiac_arrest)
            for trial in os.listdir(scenario_path):
                trial_path = os.path.join(scenario_path, trial)
                if not os.path.isdir(trial_path):
                    continue
                # trials (e.g., 0, 1, 2, s1, ...)
                for modality in os.listdir(trial_path):
                    modality_path = os.path.join(trial_path, modality)
                    if not os.path.isdir(modality_path):
                        continue
                    # modalities (e.g., audio, ego, clip_ego, clip_exo, ...)
                    for fname in os.listdir(modality_path):
                        old_path = os.path.join(modality_path, fname)
                        if not os.path.isfile(old_path):
                            continue

                        subject = subject.replace("_", "")
                        scenario = scenario.replace("_", "")

                        # build the consistent prefix
                        prefix = f"{subject}_{scenario}_{trial}_"
                        # skip if already correctly prefixed
                        if fname.startswith(prefix):
                            continue

                        # check if last folder is "exo"
                        if modality_path.endswith("exo") and fname.endswith("mp4"):
                            fname = "exo_rgb_final.mp4"

                        if modality_path.endswith("exo") and fname.endswith("hdf5"):
                            fname = "exo_rgbd_ir.hdf5"

                        if modality_path.endswith("ego") and fname.endswith("gsam2_deidentified.mp4"):
                            fname = "ego_rgb_final.mp4"

                        if modality_path.endswith("ego") and fname.endswith("720p_deidentified.mp4"):
                            fname = "ego_rgb_partial.mp4"
                        
                        if modality_path.endswith("ego") and fname.endswith("json"):
                            fname = "annotation.json"

                        if modality_path.endswith("clip_ego") and fname.endswith("npy"):
                            fname = "clip_ego_features.npy"

                        if modality_path.endswith("clip_exo") and fname.endswith("npy"):
                            fname = "clip_exo_features.npy"

                        if modality_path.endswith("i3d_rgb") and fname.endswith("npy"):
                            fname = "i3d_rgb_features.npy"

                        if modality_path.endswith("i3d_flow") and fname.endswith("npy"):
                            fname = "i3d_flow_features.npy"

                        if modality_path.endswith("resnet_ego") and fname.endswith("npy"):
                            fname = "resnet_ego_features.npy"

                        if modality_path.endswith("resnet_exo") and fname.endswith("npy"):
                            fname = "resnet_exo_features.npy"

                        if modality_path.endswith("audio") and fname.endswith("mp3"):
                            fname = "audio_deidentified.mp3"

                        if modality_path.endswith("audio") and fname.endswith("gemini_timestamped_deidentified.json"):
                            fname = "speech_transcript_deidentified.json"

                        new_fname = prefix + fname
                        new_path = os.path.join(modality_path, new_fname)
                        if os.path.exists(new_path):
                            print(f"[SKIP] target exists: {new_path}")
                            continue

                        print("-" * 20)
                        print("*" * 20)
                        print(f"[RENAME] {old_path} -> {new_path}")
                        # add this print to a text file to keep track of renamed files
                        with open("renamed_files.txt", "a") as f:
                            f.write(f"{old_path} -> {new_path}\n")
                        os.rename(old_path, new_path)



if __name__ == "__main__":
    dataset_dir = "/standard/UVA-DSA/NIST EMS Project Data/EgoEMS_AAAI2026/"
    rename_files(dataset_dir)

