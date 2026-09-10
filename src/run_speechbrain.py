import pandas as pd
import torch
import torchaudio

from speechbrain.inference.classifiers import EncoderClassifier


df = pd.read_csv("model_input.csv")

MODEL_PATH = "../models/speechbrain_lid"

language_id = EncoderClassifier.from_hparams(
    source=MODEL_PATH,
    overrides={
        "pretrained_path": MODEL_PATH
    },
    run_opts={"device": "cpu"}
)

language_id.hparams.label_encoder.ignore_len()

print("Model loaded successfully")


# predict language function 

def predict_language(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    # convert to mono

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample to 16 kHz

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=16000
        )

    with torch.no_grad():
        out_prob, score, index, text_lab = language_id.classify_batch(waveform)

    # top-1 prediction 

    raw_label = text_lab[0]
    pred_code, pred_language = raw_label.split(":", 1)
    pred_code = pred_code.strip()
    pred_language = pred_language.strip()

    # top-5 prediction 

    probs = out_prob.exp().squeeze()
    top5_probs, top5_indices = torch.topk(probs, 5)

    top5_labels = [
        language_id.hparams.label_encoder.decode_torch(i.unsqueeze(0))[0]
        for i in top5_indices
    ]

    top5_codes = []
    top5_languages = []

    for label in top5_labels:
        code, language = label.split(":", 1)
        top5_codes.append(code.strip())
        top5_languages.append(language.strip())

    return {
        "speechbrain_code": pred_code,
        "speechbrain_prediction": pred_language,
        "speechbrain_confidence": float(score.exp().item()),
        "top5_codes": top5_codes,
        "top5_languages": top5_languages,
        "top5_probabilities": [float(p) for p in top5_probs]
    }


results = []

for i, audio_path in enumerate(df["audio_path"]):

    try:
        prediction = predict_language(audio_path)
        prediction["error"] = None

    except Exception as e:
        print(f"Error on clip {i}: {e}")

        prediction = {
            "speechbrain_code": None,
            "speechbrain_prediction": None,
            "speechbrain_confidence": None,
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
            "speechbrain_checkpoint.csv",
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
    "speechbrain_predictions.csv",
    index=False,
)

print("Finished processing all clips.")
