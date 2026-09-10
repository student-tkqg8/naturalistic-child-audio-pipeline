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


def split_round1(df):
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

    return df


def split_round2(df):
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

    return df


def compile_labelled_data(split_files):
    all_labelled = []

    # load and clean all split files

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

        # Drop the same unnecessary original columns as before
        df = df.drop(
            columns=[
                df.columns[1],
                df.columns[15],
            ],
            errors="ignore"
        )

        # Keep manually labelled clips only
        labelled = df[
            df["speech_present"].notna()
        ].copy()

        if len(labelled) == 0:
            continue

        all_labelled.append(labelled)

    # ----------------------------------
    # 2. Compile full labelled dataset
    # ----------------------------------
    labelled_data = pd.concat(
        all_labelled,
        ignore_index=True
    )

    # ----------------------------------
    # 3. Add stable clip ID
    # ----------------------------------
    labelled_data["clip_id"] = range(
        1,
        len(labelled_data) + 1
    )

    # ----------------------------------
    # 4. Create langs_present
    # ----------------------------------
    language_columns = [
        "S1_lang",
        "S2_lang",
        "S3_lang"
    ]

    labelled_data["langs_present"] = (
        labelled_data[language_columns]
        .apply(
            lambda row: "; ".join(
                row.dropna()
                   .astype(str)
                   .unique()
            ),
            axis=1
        )
    )

    # Put langs_present immediately after S3_lang
    cols = list(labelled_data.columns)

    cols.remove("langs_present")

    s3_index = cols.index("S3_lang")

    cols.insert(
        s3_index + 1,
        "langs_present"
    )

    labelled_data = labelled_data[cols]

    # ----------------------------------
    # 5. Put IDs first
    # ----------------------------------
    id_columns = [
        "clip_id",
        "family_id"
    ]

    remaining_columns = [
        col for col in labelled_data.columns
        if col not in id_columns
    ]

    labelled_data = labelled_data[
        id_columns + remaining_columns
    ]

    return labelled_data


def main():
    for csv_path in round1_folder.glob("*.csv"):

        df = pd.read_csv(csv_path)
        df = split_round1(df)

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
        df = split_round2(df)

        # checks

        labelled_df = df[df["split"].notna()].copy()

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

    split_files = (
        list(round1_folder.glob("*_split.csv"))
        + list(round2_folder.glob("*_split.csv"))
    )

    labelled_data = compile_labelled_data(split_files)

    # ----------------------------------
    # 6. Create dev/eval/excluded sets
    # ----------------------------------
    dev_data = labelled_data[
        labelled_data["split"] == "dev"
    ].copy()

    eval_data = labelled_data[
        labelled_data["split"] == "eval"
    ].copy()

    excluded_data = labelled_data[
        labelled_data["split"] == "excluded"
    ].copy()

    # ----------------------------------
    # 7. Save datasets
    # ----------------------------------
    labelled_data.to_csv(
        "labelled_data_new_split.csv",
        index=False
    )

    dev_data.to_csv(
        "dev_data.csv",
        index=False
    )

    eval_data.to_csv(
        "eval_data.csv",
        index=False
    )

    excluded_data.to_csv(
        "excluded_clips.csv",
        index=False
    )

    # ----------------------------------
    # 8. Check everything
    # ----------------------------------
    print("Total labelled:", len(labelled_data))
    print("Dev:", len(dev_data))
    print("Eval:", len(eval_data))
    print("Excluded:", len(excluded_data))

    print("\nSplit proportions among retained clips:")

    retained = labelled_data[
        labelled_data["split"].isin(["dev", "eval"])
    ]

    print(
        retained["split"]
        .value_counts(normalize=True)
        .round(3)
    )

    print(
        "\nUnique clip IDs:",
        labelled_data["clip_id"].nunique()
    )

    print(
        "Duplicate file IDs:",
        labelled_data["file_id"].duplicated().sum()
    )


if __name__ == "__main__":
    main()
