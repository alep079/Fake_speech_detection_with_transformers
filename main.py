from pathlib import Path
from predict import predict

for wav_file in Path("test_audios").glob("*.wav"):
    prob, verdict = predict(str(wav_file))
    print(f"{wav_file.name}: {verdict} (prob={prob:.4f})")
