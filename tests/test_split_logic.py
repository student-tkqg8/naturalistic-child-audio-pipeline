import pandas as pd

from src.create_splits import split_round1, split_round2


def make_file_id(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"example_recording-{hours:02d}-{minutes:02d}-{seconds:02d}.wav"


def test_round1_short_run_stays_in_dev():
    df = pd.DataFrame({
        "file_id": [
            make_file_id(i * 30)
            for i in range(10)
        ]
    })

    result = split_round1(df)

    assert (result["split"] == "dev").all()


def test_round1_has_no_direct_dev_eval_boundary():
    df = pd.DataFrame({
        "file_id": [
            make_file_id(i * 30)
            for i in range(25)
        ]
    })

    result = split_round1(df)

    assert (result["split"] == "dev").sum() == 20
    assert (result["split"] == "excluded").sum() == 1
    assert (result["split"] == "eval").sum() == 4

    check = result.sort_values(
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

    direct_boundary = check[
        (check["next_start"] - check["start_seconds"] == 30)
        & (
            (
                (check["split"] == "dev")
                & (check["next_split"] == "eval")
            )
            |
            (
                (check["split"] == "eval")
                & (check["next_split"] == "dev")
            )
        )
    ]

    assert len(direct_boundary) == 0


def test_round2_pairs_do_not_cross_splits():
    df = pd.DataFrame({
        "file_id": [
            make_file_id(i * 30)
            for i in range(20)
        ],
        "unused": [None] * 20,
        "speech_present": ["yes"] * 20
    })

    result = split_round2(df)

    pair_splits = (
        result.groupby("pair_id")["split"]
        .nunique()
    )

    assert (pair_splits == 1).all()


def test_round2_split_is_reproducible():
    df = pd.DataFrame({
        "file_id": [
            make_file_id(i * 30)
            for i in range(20)
        ],
        "unused": [None] * 20,
        "speech_present": ["yes"] * 20
    })

    result_1 = split_round2(df.copy())
    result_2 = split_round2(df.copy())

    assert result_1["pair_id"].tolist() == result_2["pair_id"].tolist()
    assert result_1["split"].tolist() == result_2["split"].tolist()
