import pandas as pd

from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
)

vad_model = load_silero_vad()

df = pd.read_csv("model_input.csv")

thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

results = []

for i, audio_path in enumerate(df["audio_path"]):

    result = {}

    try:
        wav = read_audio(
            audio_path,
            sampling_rate=16000
        )

        for threshold in thresholds:

            speech_timestamps = get_speech_timestamps(
                wav,
                vad_model,
                sampling_rate=16000,
                threshold=threshold,
                return_seconds=True
            )

            suffix = f"{threshold:.2f}"

            result[f"vad_speech_detected_{suffix}"] = (
                len(speech_timestamps) > 0
            )

            result[f"vad_num_segments_{suffix}"] = (
                len(speech_timestamps)
            )

        result["vad_error"] = None

    except Exception as e:

        print(f"Error on clip {i}: {e}")

        for threshold in thresholds:
            suffix = f"{threshold:.2f}"
            result[f"vad_speech_detected_{suffix}"] = None
            result[f"vad_num_segments_{suffix}"] = None

        result["vad_error"] = str(e)

    results.append(result)

    if (i + 1) % 100 == 0:

        results_df = pd.DataFrame(results)

        checkpoint_df = pd.concat(
            [
                df.iloc[:i + 1].reset_index(drop=True),
                results_df
            ],
            axis=1
        )

        checkpoint_df.to_csv(
            "silero_all_thresholds_checkpoint.csv",
            index=False
        )

        print(f"Processed and saved {i + 1}/{len(df)} clips")


results_df = pd.DataFrame(results)

final_df = pd.concat(
    [
        df.reset_index(drop=True),
        results_df
    ],
    axis=1
)

final_df.to_csv(
    "silero_all_thresholds_predictions.csv",
    index=False
)

print("Finished processing all clips.")
