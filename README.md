# Egyptian Arabic ASR — wav2vec 2.0

End-to-end speech-to-text for **Egyptian Arabic**: scraping and auto-labelling a
dataset from YouTube, fine-tuning `wav2vec2-large-xlsr-53` with CTC, and serving
the model behind a FastAPI endpoint.

Built as a team of two — Mahmoud Haytham & Mostafa Sultan — for a speech
processing course project. Full write-up in [`docs/report.pdf`](docs/report.pdf).

## Pipeline

| # | Stage | Notebook |
|---|-------|----------|
| 01 | Pull video URLs out of YouTube playlists (`yt-dlp`, flat extraction) | [01_playlist_scrape.ipynb](notebooks/01_playlist_scrape.ipynb) |
| 02 | Download audio → strip music (Spleeter) → strip silence → chunk → auto-transcribe (Whisper `ar`) → per-video CSV | [02_dataset_pipeline.ipynb](notebooks/02_dataset_pipeline.ipynb) |
| 03 | Merge per-video CSVs, drop failed transcripts, sanity-check distributions | [03_merge_metadata.ipynb](notebooks/03_merge_metadata.ipynb) |
| 04 | EDA on the merged corpus | [04_dataset_eda.ipynb](notebooks/04_dataset_eda.ipynb) |
| 05 | CTC fine-tune wav2vec2 + WER/CER eval | [05_train_wav2vec2.ipynb](notebooks/05_train_wav2vec2.ipynb) |

The dataset was supplemented with the **Mozilla Common Voice Arabic** corpus for
annotation quality and dialect coverage (see report §2.2).

## Dataset

`data/combined.csv` — ~29.9k labelled utterances:

| Field | Values |
|-------|--------|
| `Audio Path` | wav chunk filename |
| `Transcript` | Egyptian-Arabic transcript (Whisper pseudo-labels, normalized) |
| `Gender` | male 86% / female 14% — known imbalance |
| `Age` / `Mood` / `Accent` | adult · mostly calm · Egyptian Arabic |

`data/vocab.json` — the 50-token character vocabulary built from the normalized
transcripts (arabic letters + digits + CTC special tokens), ready for the
training notebook and the tokenizer in `serve/`.

Sources: movies, street interviews, and radio interviews collected from YouTube.
Audio itself isn't committed — notebooks 01–03 reproduce it; labels are
machine-generated so expect a non-trivial error rate.

## Results

4 epochs, `wav2vec2-large` on Kaggle GPU (from the report):

| Epoch | Train loss | Val loss | WER | CER |
|-------|-----------|----------|-----|-----|
| 1 | 2.49 | 2.24 | 0.78 | 0.97 |
| 4 | 0.71 | 22.96 | **0.57** | 0.96 |

WER kept improving while val loss diverged — a classic CTC artifact — but
0.57 WER is still far from production quality. Honest limitations: small noisy
pseudo-labelled corpus, heavy gender skew, Whisper label noise, and no language
model rescoring. These are first-pass results, not a finished product.

## Serving

```bash
pip install -r requirements.txt

cd serve
MODEL_DIR=/path/to/saved/model uvicorn app:app --port 8000

# another terminal
python client.py my_audio.wav
# → {"transcription": "..."}
```

`POST /transcribe/` accepts any ffmpeg-decodable audio, resamples to 16 kHz mono,
runs the CTC model, and returns the decoded transcript. `GET /health` for a
liveness check. The model directory is produced by the last cell of notebook 05
(`model.save_pretrained(OUTPUT_DIR)`).

## Extras

Classical speech-DSP analyses (same course, earlier assignments) under
[`extras/`](extras/):

- `01_spectrogram_formant_analysis.ipynb` — STFT spectrograms, wideband vs
  narrowband resolution, F0 tracking, formant structure of vowels vs
  instruments, male/female voice comparison.
- `02_autocorrelation_dft_dsp.ipynb` — autocorrelation and DFT/IDFT implemented
  from scratch, Wiener–Khinchin via the spectrum, noise robustness, vowel vs
  consonant analysis.

## Repo layout

```
├── data/                combined.csv corpus metadata + vocab.json
├── notebooks/           the pipeline: scrape → build dataset → EDA → train
├── serve/               FastAPI transcription server + test client
├── extras/              classical DSP notebooks (+ their audio assets)
└── docs/report.pdf      full project report
```

## Credits

- **Mahmoud Haytham** — [github.com/mahmoudhaytham25](https://github.com/mahmoudhaytham25)
- **Mostafa Sultan** — teammate

Mozilla Common Voice Arabic corpus · Whisper pseudo-labelling · Spleeter source
separation · HuggingFace `wav2vec2` checkpoints.
