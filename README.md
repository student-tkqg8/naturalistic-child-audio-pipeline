# Evaluation of Pretrained Models for Automatic Language Annotation of Naturalistic Child-Centred Audio

This repository contains the code used for an MSc dissertation evaluating pretrained spoken language identification (LID) models on naturalistic child-centred daylong audio.

The project compares four pretrained LID systems and investigates the use of voice activity detection (VAD) and confidence-based filtering as part of a practical automatic annotation pipeline.

## Models

The following pretrained models were evaluated:

- MMS-LID-256
- SpeechBrain VoxLingua107 ECAPA-TDNN
- Whisper small
- Whisper medium

Silero VAD was additionally evaluated for speech detection and used in the final VAD-LID pipeline.

## Repository structure

```text
.
├── src/
│   ├── create_splits.py
│   ├── run_mms.py
│   ├── run_speechbrain.py
│   ├── run_whisper_small.py
│   ├── run_whisper_medium.py
│   ├── run_silero_vad.py
│   └── run_vad_whisper.py
│
├── analysis/
│   ├── 01_model_comparison.ipynb
│   ├── 02_error_and_confidence_analysis.ipynb
│   ├── 03_multilingual_and_family_analysis.ipynb
│   ├── 04_vad_and_pipeline_analysis.ipynb
│   └── 05_norwegian_error_analysis.ipynb
│
├── data/
│   ├── README.md
│   └── example_model_input.csv
│
├── tests/
│   └── test_split_logic.py
│
├── requirements.txt
└── .gitignore
```

## Data

The original audio recordings and annotation files are not included in this repository because they contain sensitive participant data and cannot be publicly redistributed.

`data/example_model_input.csv` contains entirely synthetic data demonstrating the structure of the metadata used by the model and analysis scripts. All identifiers, paths and observations in this file are fictional.

The recordings used in the study were divided into 30-second clips and accompanied by clip-level annotations including speech presence, speech type and language information.

See [`data/README.md`](data/README.md) for further information.

## Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

Whisper also requires FFmpeg to be installed separately and available on the system PATH.

Model weights are not included in this repository and should be obtained from their respective model providers.

## Model inference

The scripts in `src/` contain the inference procedures used for each model:

```text
run_mms.py
run_speechbrain.py
run_whisper_small.py
run_whisper_medium.py
```

`run_silero_vad.py` evaluates Silero VAD at a range of speech-detection thresholds.

`run_vad_whisper.py` implements the final VAD-LID pipeline using Silero VAD together with Whisper medium and confidence-based filtering.

The example metadata in `data/example_model_input.csv` illustrates the expected input structure. Audio paths must be replaced with paths to locally available audio files before inference can be run.

## Data splitting

`src/create_splits.py` contains the procedures used to construct the development and held-out evaluation sets.

For recordings with long contiguous annotated sequences, temporal blocks were used to reduce leakage between development and evaluation data. For data sampled as adjacent clip pairs, assignment was performed at pair level so that members of the same pair could not occur in different splits.

A fixed random seed is used for reproducibility.

Basic tests of the split logic are provided in `tests/test_split_logic.py`.

Tests can be run from the repository root using:

```bash
pytest
```

## Analysis

The notebooks in `analysis/` reproduce the principal analyses used in the dissertation.

- `01_model_comparison.ipynb` — comparison of the four pretrained LID models, including overall and per-language performance and pairwise statistical comparisons.
- `02_error_and_confidence_analysis.ipynb` — analysis of Whisper-medium prediction errors and confidence.
- `03_multilingual_and_family_analysis.ipynb` — multilingual and family-level analyses.
- `04_vad_and_pipeline_analysis.ipynb` — VAD threshold analysis and evaluation of the combined VAD-LID pipeline.
- `05_norwegian_error_analysis.ipynb` — analysis of recurrent Norwegian-language predictions.

The notebooks included in the public repository do not contain participant-level outputs.

## Evaluation

The core model comparison evaluates LID performance on live-speech clips containing a single valid reference language supported by the relevant models.

Additional analyses examine:

- performance across languages;
- multilingual recordings;
- variation across families;
- model confidence and errors;
- speech/no-speech discrimination;
- VAD threshold selection; and
- the combined VAD-LID annotation pipeline.

Because Whisper represents Mandarin using a single Chinese language token and does not provide Cantonese as a distinct LID class, model-specific label handling is applied during evaluation.

## Reproducibility and privacy

No raw recordings, participant metadata, original annotation files, participant identifiers, or clip-level prediction files are distributed in this repository.

The public example data are synthetic. Users with authorised access to the original dataset can adapt the example metadata structure to reproduce the analyses.

Model predictions and other generated files are excluded from version control to prevent accidental release of participant-linked information.
