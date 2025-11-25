import os
import pandas as pd
import argparse



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to MVC folder containing the 'en' folder")
    args = ap.parse_args()

    mozillaPath = f'{args.path}/en/validated.tsv'
    audioPath   = f'{args.path}/en/clips'
    cvDelta = pd.read_csv(mozillaPath, sep='\t')

    print("Helper script for pruning the absolutely enormous MVC dataset down to save space and time.")

    # print(cvDelta.head)
    # print(cvDelta.columns)
    # print(len(cvDelta))
    cvDelta = cvDelta.dropna(subset=['age', 'gender', 'accents'])


    filesToKeep = set(cvDelta["path"].apply(os.path.basename))
    allClips = set(os.listdir(audioPath))
    forDeletion = allClips - filesToKeep


    #iterate through clips folder and delete files not in the keeping set
    
    print(f"There are {len(allClips)} clips in {mozillaPath}, {len(filesToKeep)} of which have age, gender, and accents info.")
    print(f"Doing this will delete {len(forDeletion)} clips to free up space.")
    print("Press ENTER to continue and delete files, OR:")
    print("Press CTRL+C or type anything then ENTER to cancel.")
    user_input = input("> ")

    if user_input.strip() != "":
        print("Cancelled.")
        exit(0)

    # delete audio files and free up some space
    deletedCount = 0
    for filename in forDeletion:
        os.remove(os.path.join(audioPath, filename))
        deletedCount += 1

    print(f"Deleted {deletedCount} files.")
    


