import os
import tempfile

import librosa
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Directory containing the fine-tuned model + processor saved by
# notebooks/05_train_wav2vec2.ipynb (model.save_pretrained(OUTPUT_DIR)).
MODEL_DIR = os.environ.get("MODEL_DIR", "./wav2vec2-egyptian-arabic")

app = FastAPI(title="Egyptian Arabic ASR")

model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model.eval()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe/")
async def transcribe(audio_file: UploadFile = File(...)):
    # torchaudio needs a seekable file path — spool the upload to a temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await audio_file.read())
            tmp_path = tmp.name
        audio, sampling_rate = torchaudio.load(tmp_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode audio file")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Resample to 16 kHz mono
    audio = audio[0].numpy()
    if sampling_rate != 16000:
        audio = librosa.resample(np.asarray(audio), orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000

    input_values = processor(
        audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)[0]
    return {"transcription": transcription}
