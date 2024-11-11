from transformers import *
import torch
import soundfile as sf
# import librosa
import os
import torchaudio

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# wav2vec2_model_name = "facebook/wav2vec2-base-960h" # 360MB
wav2vec2_model_name = "facebook/wav2vec2-large-960h-lv60-self" # pretrained 1.26GB
# wav2vec2_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english" # English-only, 1.26GB
# wav2vec2_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic" # Arabic-only, 1.26GB
# wav2vec2_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish" # Spanish-only, 1.26GB


# whisper_model_name = "openai/whisper-tiny.en" # English-only, ~ 151 MB
# whisper_model_name = "openai/whisper-base.en" # English-only, ~ 290 MB
# whisper_model_name = "openai/whisper-small.en" # English-only, ~ 967 MB
# whisper_model_name = "openai/whisper-medium.en" # English-only, ~ 3.06 GB
# whisper_model_name = "openai/whisper-tiny" # multilingual, ~ 151 MB
# whisper_model_name = "openai/whisper-base" # multilingual, ~ 290 MB
# whisper_model_name = "openai/whisper-small" # multilingual, ~ 967 MB
whisper_model_name = "openai/whisper-medium" # multilingual, ~ 3.06 GB
# whisper_model_name = "openai/whisper-large-v2" # multilingual, ~ 6.17 GB

wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_name).to(device)
def load_audio(audio_path):
  """Load the audio file & convert to 16,000 sampling rate"""
  # load our wav file
  speech, sr = torchaudio.load(audio_path)
  resampler = torchaudio.transforms.Resample(sr, 16000)
  speech = resampler(speech)
  return speech.squeeze()

def get_transcription_wav2vec2(audio_path, model, processor):
  speech = load_audio(audio_path)
  input_features = processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"].to(device)
  # perform inference
  logits = model(input_features)["logits"]
  # use argmax to get the predicted IDs
  predicted_ids = torch.argmax(logits, dim=-1)
  transcription = processor.batch_decode(predicted_ids)[0]
  return transcription.lower()

# initialize the pipeline
pipe = pipeline("automatic-speech-recognition", 
                model=whisper_model_name, device=device)

def get_long_transcription_whisper(audio_path, pipe, return_timestamps=True, chunk_length_s=10, stride_length_s=2):
    """Get the transcription of a long audio file using the Whisper model"""
    return pipe(load_audio(audio_path).numpy(), return_timestamps=return_timestamps,
                  chunk_length_s=chunk_length_s, stride_length_s=stride_length_s)

output = get_long_transcription_whisper(
    "rep/SLI0-4.wav", 
    pipe, chunk_length_s=10, stride_length_s=1)

print(output["text"])

output = get_long_transcription_whisper(
    "rep/SLI10-1.wav", 
    pipe, chunk_length_s=10, stride_length_s=1)

print(output["text"])

output = get_long_transcription_whisper(
    "rep/SLI11-90.wav", 
    pipe, chunk_length_s=10, stride_length_s=1)

print(output["text"])

