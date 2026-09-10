import pandas as pd

# import whisper medium

import whisper 
import torch
from whisper.tokenizer import LANGUAGES

df = pd.read_csv("model_input.csv")

model = whisper.load_model("medium", device="cpu")

# predict function for whisper medium

def predict_language_whisper_med(audio_path): 

    # ffmpeg used to decode file, no resampling needed

    audio = whisper.load_audio(str(audio_path))

    # make exactly 30s

    audio = whisper.pad_or_trim(audio)


    mel = whisper.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels
    ).to(model.device)

    with torch.no_grad():
        _, probs = model.detect_language(mel)

    # top-5 predictions
    
    top5 = sorted(
        probs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    top5_codes = [
        code
        for code, prob in top5
    ]

    top5_languages = [
        LANGUAGES.get(code, code).title()
        for code in top5_codes
    ]

    top5_probabilities = [
        float(prob)
        for code, prob in top5
    ]

    return {
        "whisper_med_code": top5_codes[0],
        "whisper_med_prediction": top5_languages[0],
        "whisper_med_confidence": top5_probabilities[0],
        "top5_codes": top5_codes,
        "top5_languages": top5_languages,
        "top5_probabilities": top5_probabilities,
    }


results = []

for i, audio_path in enumerate(df["audio_path"]):

    try:
        prediction = predict_language_whisper_med(audio_path)
        prediction["error"] = None

    except Exception as e:
        print(f"Error on clip {i}: {e}")

        prediction = {
            "whisper_med_code": None,
            "whisper_med_prediction": None,
            "whisper_med_confidence": None,
            "top5_codes": None,
            "top5_languages": None,
            "top5_probabilities": None,
            "error": str(e),
        }

    results.append(prediction)

    if (i + 1) % 100 == 0:

        results_df = pd.DataFrame(results)

        checkpoint_df = pd.concat(
            [
                df.iloc[:i + 1].reset_index(drop=True),
                results_df,
            ],
            axis=1,
        )

        checkpoint_df.to_csv(
            "whisper_medium_checkpoint.csv",
            index=False,
        )

        print(f"Processed and saved {i + 1}/{len(df)} clips")


# save final results

results_df = pd.DataFrame(results)

final_df = pd.concat(
    [
        df.reset_index(drop=True),
        results_df,
    ],
    axis=1,
)

final_df.to_csv(
    "whisper_medium_predictions.csv",
    index=False,
)

print("Finished processing all clips.")
