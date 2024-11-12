from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
import torch
import os
import torchaudio
# from newestModel import insertIntoCSV

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

pipe = pipeline("automatic-speech-recognition",
                model=whisper_model_name, device=device)

def get_long_transcription_whisper(audio_path, pipe, return_timestamps=True, chunk_length_s=10, stride_length_s=2):
    """Get the transcription of a long audio file using the Whisper model"""
    return pipe(load_audio(audio_path).numpy(), return_timestamps=return_timestamps,
                  chunk_length_s=chunk_length_s, stride_length_s=stride_length_s)

def slidingWindow(text):
  text = text.strip()
  text = text.lower()
  word_mapper = {}
  replacement = ["'",",","."]
  for r in replacement:
    text = text.replace(r, "")

  new_text = text.split(" ")
  for i, words in enumerate(new_text):
    if words not in word_mapper:
      word_mapper[words] = [i]
    else:
      # if the previous same found word was said right before -> stutter
      if word_mapper[words][-1] == i - 1:
        return True
      elif i - word_mapper[words][-1] < 5: # Here im saying if the same word was said within a range of 5 words -> might need to change this later but idk
        # if there was a word repeated -> it checks if the next word is the same (sentences like "And that was and that was" should also be considered as stutters)
        if i+1 < len(new_text):
          if new_text[i+1] in word_mapper and word_mapper[new_text[i+1]][-1] == word_mapper[words][-1] + 1:
            return True
      else:
         word_list = word_mapper[words]
         word_list.append(i)
         word_mapper[words] = word_list
  return False

def didStutter(audio_file, pipe = pipe, chunk_length_s=10, stride_length_s=1):
  
  for chunks in range(7,11):
    output = get_long_transcription_whisper(audio_file, pipe, chunk_length_s=chunks, stride_length_s=stride_length_s)
    print("Output:",output["text"])
    if slidingWindow(output["text"]):
      print("STUTTER")
      print()
      return True
  print("NO STUTTER")
  print()
  return False

directory_path = r"C:/Users/andrew/Downloads/ml_stuttering/clips/stuttering-clips/clips"
all_entries = os.listdir(directory_path)

stutter = 0
no_stutter = 0

testFiles = [r"FluencyBank_010_4.wav",r"FluencyBank_010_79.wav",r"FluencyBank_010_82.wav",r"FluencyBank_010_97.wav", ]

for filename in testFiles:#all_entries:
    print()
    f = filename.strip().split("_")
    print(filename)
    path = os.path.join(directory_path, filename)
    print("Path", path)
    if didStutter(path):
        d = [f[0],f[1],f[2][:-4], "YES"]
        # insertIntoCSV(d)
        stutter += 1
    else:
        d = [f[0],f[1],f[2][:-4], "NO"]
        # insertIntoCSV(d)
        no_stutter += 1
