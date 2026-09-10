import librosa
import pandas as pd
import pycountry
import torch

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


df = pd.read_csv("model_input.csv")

MODEL_PATH = "../models/mms-lid-256"

# load model

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model.eval()


def load_audio(audio_path):
    waveform, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )
    return waveform, sample_rate


# add all language codes using pycountry 

mms_custom_codes = {
    "cmn": "Mandarin",
    "yue": "Cantonese", 
    "zlm": "Malay",
    "kmr": "Northern Kurdish", 
    "ckb": "Central Kurdish",
    "pan": "Punjabi",
}

def code_to_label(code): 
    if code in mms_custom_codes: 
        return mms_custom_codes[code]

    language = pycountry.languages.get(alpha_3=code)

    if language is not None: 
        return language.name

    return code


# new predict function to include top5, add all lang codes and round probabilities

def predict_language(audio_path):
    waveform, sample_rate = load_audio(audio_path)

    inputs = feature_extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)

    top_probs, top_ids = torch.topk(probabilities, k=5)

    top_codes = [
        model.config.id2label[i.item()]
        for i in top_ids[0]
    ]

    top_languages = [
        code_to_label(code)
        for code in top_codes
    ]

    top_probabilities = [
        round(p, 4)
        for p in top_probs[0].tolist()
    ]

    return {
        "mms_code": top_codes[0],
        "mms_prediction": top_languages[0],
        "mms_confidence": top_probabilities[0],
        "top5_codes": top_codes,
        "top5_languages": top_languages,
        "top5_probabilities": top_probabilities,
    }


# full dev set run 

results = []

for i, audio_path in enumerate(df["audio_path"]):

    prediction = predict_language(audio_path)
    results.append(prediction)

    if (i + 1) % 100 == 0:

        results_df = pd.DataFrame(results)

        checkpoint_df = pd.concat(
            [df.iloc[:i + 1].reset_index(drop=True), results_df],
            axis=1,
        )

        checkpoint_df.to_csv(
            "mms_checkpoint.csv",
            index=False,
        )

        print(f"Processed and saved {i + 1}/{len(df)} clips")

# save final results

results_df = pd.DataFrame(results)

final_df = pd.concat(
    [df.reset_index(drop=True), results_df],
    axis=1,
)

final_df.to_csv(
    "mms_predictions.csv",
    index=False,
)

print("Finished processing all clips.")
