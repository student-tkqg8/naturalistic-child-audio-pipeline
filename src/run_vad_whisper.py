import pandas as pd


# selected development-set thresholds

VAD_THRESHOLD = 0.20
CONF_THRESHOLD = 0.70

vad = pd.read_csv("silero_all_thresholds_predictions.csv")
whisper = pd.read_csv("whisper_medium_predictions.csv")

whisper_cols = (
    whisper[
        [
            "file_id",
            "whisper_med_prediction",
            "whisper_med_confidence"
        ]
    ]
    .drop_duplicates("file_id")
)

df = vad.merge(
    whisper_cols,
    on="file_id",
    how="left"
)

df["split_clean"] = (
    df["split"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["speech_true"] = (
    df["speech_present"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["pred_language"] = (
    df["whisper_med_prediction"]
    .astype("string")
    .str.strip()
    .str.lower()
)

def parse_languages(x):
    if pd.isna(x):
        return set()

    langs = {
        str(lang).strip().lower()
        for lang in str(x).split(";")
        if str(lang).strip()
    }

    langs -= {"neutral", "mixed", "unclear"}

    # map Mandarin reference labels to Whisper's Chinese label
  
    return {
        "chinese" if lang == "mandarin" else lang
        for lang in langs
    }

df["true_language_set"] = (
    df["langs_present"]
    .apply(parse_languages)
)

df["n_valid_languages"] = (
    df["true_language_set"]
    .map(len)
)

vad_col = f"vad_speech_detected_{VAD_THRESHOLD:.2f}"

df["vad_pass"] = (
    df[vad_col]
    .astype("string")
    .str.strip()
    .str.lower()
    .map({
        "true": True,
        "false": False
    })
)

# exclude electronic speech from deployment analysis

eligible = (
    df["speech_true"].eq("no")
    |
    (
        df["speech_true"].eq("yes")
        & df["n_valid_languages"].ge(1)
        & ~df["speech_type"]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("electronic")
    )
)

analysis = df[
    eligible
    & df["vad_pass"].notna()
].copy()

analysis["vad_pass"] = (
    analysis["vad_pass"]
    .astype(bool)
)

analysis["single_language"] = (
    analysis["speech_true"].eq("yes")
    & analysis["n_valid_languages"].eq(1)
)

analysis["multilingual"] = (
    analysis["speech_true"].eq("yes")
    & analysis["n_valid_languages"].ge(2)
)

analysis["language_matches_any_reference"] = (
    analysis.apply(
        lambda r:
            r["pred_language"] in r["true_language_set"],
        axis=1
    )
)

analysis["chinese_prediction"] = (
    analysis["pred_language"].eq("chinese")
)

analysis["auto_no_speech"] = (
    ~analysis["vad_pass"]
)

# automatically accept language predictions above the confidence threshold

analysis["auto_language"] = (
    analysis["vad_pass"]
    & ~analysis["chinese_prediction"]
    & analysis["whisper_med_confidence"].ge(CONF_THRESHOLD)
)

# chinese predictions sent to manual review, as Whisper does not distinguish Mandarin and Cantonese

analysis["manual_review"] = (
    analysis["vad_pass"]
    & (
        analysis["chinese_prediction"]
        | analysis["whisper_med_confidence"].lt(CONF_THRESHOLD)
    )
)

analysis.to_csv(
    "vad_whisper_pipeline.csv",
    index=False
)
