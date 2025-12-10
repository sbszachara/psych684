import argparse
import os
import pandas as pd
import shutil

# helper script for making a smaller zip to share

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to root MVC folder containing the 'en' folder")
    args = ap.parse_args()

    mozillaPath = f'{args.path}/en/validated.tsv'
    audioPath   = f'{args.path}/en/clips'
    cvDelta = pd.read_csv(mozillaPath, sep='\t')

    # print(cvDelta.head)
    # print(cvDelta.columns)
    # print(len(cvDelta))
    cvDelta = cvDelta.dropna(subset=['age', 'gender', 'accents'])
    print(len(cvDelta))

    # if the tsv is huge, grab a more reasonable sample
    # there are over 900,000 clips in MCV even after the above dropna so this is necessary.
    #cvDelta = cvDelta.sample(n=min(args.sample_size, len(cvDelta)), random_state=args.seed)

    print("check just for US English:")
    cvDelta = cvDelta.loc[cvDelta['accents']=="United States English"]

    print(len(cvDelta))
    print("Demographic info:")
    print(cvDelta['gender'].value_counts())
    print((cvDelta['gender'].value_counts(normalize=True) * 100).round(6))


    #cvDelta['path'] = cvDelta['path'].apply(lambda x: os.path.join(audioPath, x))