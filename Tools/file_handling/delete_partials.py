#!/usr/bin/env python3
import os
import argparse

def process_ego_dirs(root, dry_run=False):
    """
    Walk root, find each 'ego' folder,
    and delete *_rgb_partial.mp4 only if *_rgb_final.mp4 also exists.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) != 'ego':
            continue

        final_files   = [f for f in filenames if f.endswith('_rgb_final.mp4')]
        partial_files = [f for f in filenames if f.endswith('_rgb_partial.mp4')]

        # only proceed when both exist
        if not final_files or not partial_files:
            continue

        print(f"\nFound in: {dirpath}")
        print("  Final:")
        for f in final_files:
            print("    ", f)
        print("  Partial:")
        for f in partial_files:
            print("    ", f)

        if dry_run:
            print("  [dry-run] skipping deletion")
            continue

        for pf in partial_files:
            path = os.path.join(dirpath, pf)
            try:
                os.remove(path)
                print("    Deleted:", pf)
            except Exception as e:
                print("    ERROR deleting", pf, "→", e)


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Remove ego_rgb_partial.mp4 when an ego_rgb_final.mp4 exists"
    )
    p.add_argument("root", help="Dataset root directory")
    p.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="List files but do not delete"
    )
    args = p.parse_args()
    process_ego_dirs(args.root, dry_run=args.dry_run)
