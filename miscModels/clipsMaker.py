from pydub import AudioSegment
from pydub.utils import make_chunks
import os

def makeClips(frontPath: str,audioFile: str, chunk_length: int): #chunk_length in ms
    myaudio = AudioSegment.from_file(frontPath+ "/" +audioFile , "wav") 
    chunks = make_chunks(myaudio, chunk_length) #Make chunks of one sec
    file, name = frontPath + "clips/", audioFile.strip('.wav')
    
    for i, chunk in enumerate(chunks):
        chunk_name = file + name + "-{0}.wav".format(i)
        chunk.export(chunk_name, format="wav")

# rep\SLI14-4.wav
# makeClips("rep\SLI14-4.wav", 3000)

def createAllClips():
    # C:\Users\andrew\Desktop\stutter_model\rep
    # C:\Users\andrew\Desktop\stutter_model\norep
    [makeClips("norep",f, 3000) for f in os.listdir("C:/Users/andrew/Desktop/stutter_model/norep")]
    [makeClips("rep",f, 3000) for f in os.listdir("C:/Users/andrew/Desktop/stutter_model/rep")] 
    print("Created all")
    return

#"C:\Users\andrew\Downloads\ml_stuttering\clips\stuttering-clips\clips"

createAllClips()