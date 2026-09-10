from pathlib import Path 

import numpy as np
import pandas as pd

import re


round1_folder = Path("round1_family_csvs")
round2_folder = Path("round2_family_csvs")

RANDOM_SEED = 42


def parse_clip(path):
    name = Path(str(path).replace("\\", "/")).name

    match = re.match(
        r"(.+)-(\d{2})-(\d{2})-(\d{2})\.wav$",
        name
    )

    if match is None:
        return pd.Series([None, None])

    recording_id = match.group(1)

    h = int(match.group(2))
    m = int(match.group(3))
    s = int(match.group(4))

    start_seconds = h * 3600 + m * 60 + s

    return pd.Series([recording_id, start_seconds])


for csv_path in round1_folder.glob("*.csv"):

    df = pd.read_csv(csv_path)
    file_col = df.columns[0]


    # extract recording and time
    
    df[["recording_id", "start_seconds"]] = df[file_col].apply(parse_clip)

    # identify contiguous runs
 
    df = df.sort_values(
        ["recording_id", "start_seconds"]
    ).reset_index(drop=True)

    df["gap"] = (
        df.groupby("recording_id")["start_seconds"]
          .diff()
    )

    df["new_run"] = (
        df["gap"].isna()
        | (df["gap"] != 30)
    )

    df["run_number"] = (
        df.groupby("recording_id")["new_run"]
          .cumsum()
          .astype(int)
    )

    df["run_id"] = (
        df["recording_id"].astype(str)
        + "_run_"
        + df["run_number"].astype(str)
    )


    # split each run

    df["split"] = None

    MIN_RUN_SIZE = 20

    for run_id, run in df.groupby("run_id"):

        idx = run.index
        n = len(run)

        # very short runs kept entirely in dev split
      
        if n < MIN_RUN_SIZE:
            df.loc[idx, "split"] = "dev"
            continue

        # approximate 80% boundary
      
        boundary = int(n * 0.8)

        # first 80% goes to dev split
      
        df.loc[idx[:boundary], "split"] = "dev"

        # exclude one clip at the boundary 
      
        df.loc[idx[boundary], "split"] = "excluded"

        # remaining clips go to eval split
      
        df.loc[idx[boundary + 1:], "split"] = "eval"


    # show results
  
    print(df["split"].value_counts())
    print()
    print(df["split"].value_counts(normalize=True))


    # check for leakage

    check = df.sort_values(
        ["recording_id", "start_seconds"]
    ).copy()

    check["next_start"] = (
        check.groupby("recording_id")["start_seconds"]
             .shift(-1)
    )

    check["next_split"] = (
        check.groupby("recording_id")["split"]
             .shift(-1)
    )

    leaks = check[
        (check["next_start"] - check["start_seconds"] == 30)
        &
        (
            ((check["split"] == "dev") & (check["next_split"] == "eval"))
            |
            ((check["split"] == "eval") & (check["next_split"] == "dev"))
        )
    ]

    print()
    print("Contiguous dev/eval boundaries:", len(leaks))

    # save

    output_path = csv_path.with_name(
        csv_path.stem + "_split.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")


for csv_path in round2_folder.glob("*.csv"):

    # skip files already split
  
    if csv_path.stem.endswith("_split"):
        continue

    print(f"\nProcessing: {csv_path.name}")

    df = pd.read_csv(csv_path)

    file_col = df.columns[0]
    speech_present_col = df.columns[2]

    # add empty columns to full dataframe
  
    df["recording_id"] = None
    df["start_seconds"] = np.nan
    df["pair_id"] = None
    df["split"] = None

    # keep only manually labelled clips
  
    labelled_mask = df[speech_present_col].notna()

    labelled_df = df.loc[labelled_mask].copy()

    # extract recording and time for labelled clips
  
    labelled_df[["recording_id", "start_seconds"]] = (
        labelled_df[file_col].apply(parse_clip)
    )

    # sort labelled clips chronologically
  
    labelled_df = labelled_df.sort_values(
        ["recording_id", "start_seconds"]
    ).copy()

    # create pair IDs
  
    labelled_df["pair_id"] = None

    for recording_id, rec in labelled_df.groupby(
        "recording_id",
        sort=False
    ):

        indices = rec.index.tolist()
        pair_num = 1
        i = 0

        while i < len(indices):

            current_idx = indices[i]

            if i + 1 < len(indices):

                next_idx = indices[i + 1]

                gap = (
                    labelled_df.loc[next_idx, "start_seconds"]
                    - labelled_df.loc[current_idx, "start_seconds"]
                )

                # two clips exactly 30 seconds apart = pair
              
                if gap == 30:

                    pair_id = (
                        f"{recording_id}_pair_{pair_num}"
                    )

                    labelled_df.loc[current_idx, "pair_id"] = pair_id
                    labelled_df.loc[next_idx, "pair_id"] = pair_id

                    pair_num += 1
                    i += 2
                    continue

            # any unpaired labelled clip gets its own group
          
            single_id = (
                f"{recording_id}_single_{pair_num}"
            )

            labelled_df.loc[current_idx, "pair_id"] = single_id

            pair_num += 1
            i += 1

    # random 80/20 split at pair level
  
    pairs = labelled_df["pair_id"].unique().copy()

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(pairs)

    n_eval = round(len(pairs) * 0.20)

    eval_pairs = set(pairs[:n_eval])

    labelled_df["split"] = np.where(
        labelled_df["pair_id"].isin(eval_pairs),
        "eval",
        "dev"
    )

    # write results back into full corpus

    df.loc[labelled_df.index, "recording_id"] = (
        labelled_df["recording_id"]
    )

    df.loc[labelled_df.index, "start_seconds"] = (
        labelled_df["start_seconds"]
    )

    df.loc[labelled_df.index, "pair_id"] = (
        labelled_df["pair_id"]
    )

    df.loc[labelled_df.index, "split"] = (
        labelled_df["split"]
    )


    # checks

    pair_check = (
        labelled_df.groupby("pair_id")["split"].nunique()
    )

    print("Labelled clips:", len(labelled_df))
    print()

    print(labelled_df["split"].value_counts())
    print()

    print(
        labelled_df["split"]
        .value_counts(normalize=True)
        .round(3)
    )

    print(
        "Pairs split across dev/eval:",
        (pair_check > 1).sum()
    )

    print(
        "Unlabelled clips left unsplit:",
        df["split"].isna().sum()
    )

    # save

    output_path = csv_path.with_name(
        csv_path.stem + "_split.csv"
    )

    df.to_csv(output_path, index=False)

    print("Saved:", output_path)


all_labelled = []

# load and clean all split files

split_files = (
    list(round1_folder.glob("*_split.csv"))
    + list(round2_folder.glob("*_split.csv"))
)

for file in split_files:

    df = pd.read_csv(file)

    # rename annotation columns
  
    df = df.rename(columns={
        df.columns[0]: "file_id",
        df.columns[2]: "speech_present",
        df.columns[3]: "speech_type",
        df.columns[4]: "num_speakers",
        df.columns[5]: "S1_identity",
        df.columns[6]: "S2_identity",
        df.columns[7]: "S3_identity",
        df.columns[8]: "S1_addressee",
        df.columns[9]: "S2_addressee",
        df.columns[10]: "S3_addressee",
        df.columns[11]: "num_langs",
        df.columns[12]: "S1_lang",
        df.columns[13]: "S2_lang",
        df.columns[14]: "S3_lang",
        df.columns[16]: "child_vocalising",
    })

    if len(df.columns) > 17:
        df = df.rename(columns={
            df.columns[17]: "comments"
        })

    # family ID from filename
  
    df["family_id"] = file.stem.replace("_split", "")
