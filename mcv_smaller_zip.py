import argparse
import os
import pandas as pd
import shutil

# helper script for making a smaller zip to share

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to root MVC folder containing the 'en' folder")
    ap.add_argument("--sample-size", type=int, default=10000, help="How many samples to copy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="subset_en", help="Output folder for the mini dataset")
    args = ap.parse_args()


    tsv_path = os.path.join(args.path, "en", "validated.tsv")
    clips_path = os.path.join(args.path, "en", "clips")

    out_root = args.output
    out_clips = os.path.join(out_root, "clips")
    out_tsv = os.path.join(out_root, "validated.tsv")

    os.makedirs(out_clips, exist_ok=True)

    print("Loading TSV...")
    df = pd.read_csv(tsv_path, sep="\t")

    print("Dropping rows missing demographic fields...")
    df = df.dropna(subset=["age", "gender", "accents"])

    print(f"Dataset size after filtering: {len(df)} rows")

    # Sample the dataset
    sample_n = min(args.sample_size, len(df))
    df = df.sample(n=sample_n, random_state=args.seed)

    print(f"Selected {len(df)} samples")

    # Path → absolute path for copying
    df["src_path"] = df["path"].apply(lambda x: os.path.join(clips_path, x))
    df["dst_path"] = df["path"].apply(lambda x: os.path.join(out_clips, x))

    print("Copying audio files...")

    copied = 0
    missing = 0

    for src, dst in zip(df["src_path"], df["dst_path"]):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

    print(f"Copied: {copied}")
    print(f"Missing: {missing}")

    # Save the trimmed TSV
    df.drop(columns=["src_path", "dst_path"], inplace=True)
    df.to_csv(out_tsv, sep="\t", index=False)

    print(f"Subset dataset created at: {out_root}")