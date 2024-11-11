import pandas as pd
import librosa
import librosa.feature
import soundfile as sf
import json
import ijson
import random
import math
import numpy as np
import os 

def mel_spectrogram(y, sr):
    mel_spect = librosa.feature.melspectrogram(y=y,sr=sr, n_mels=64)
    return mel_spect

def mfcc_converter(y,sr, n_fft=2048, hop_length=512, n_mfcc=13):
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
#   print(mfcc.shape)
  mfcc = mfcc.T
  return mfcc

def file_to_df(csv, JSON_PATH="C:/Users/andrew/Downloads/data.json", save=False): # extracts all the downloaded wav data and returns
    SAMPLERATE = 16000
    # SPECSHAPE = (64,94)
    # MFCCSHAPE = (13,)
     
    data = {
        "Prolongation":[],
        "WordRep":[],
        "SoundRep":[],
        "Block":[],
        "Interjection":[],
        "Stutter":[],
        "spect":[],
        "mfcc":[],
        "natural":[],
        "total":[] #0 -> Stutter, 1 -> Prolongation, 2 -> WordRep, 3 -> SoundRep, 4 -> Block, 5 -> Interjection
    }
    #/content/drive/MyDrive/stuttering_clips/clips/stuttering-clips/clips/FluencyBank_010_0.wav
    # new_df = pd.DataFrame(columns=csv_df.columns)
    i = 0
    count = 0
    for index, row in csv.iterrows():
    # find each wav file
        
        path_name = f'{row["Show"]}_{row["EpId"]}_{row["ClipId"]}.wav'
        "C:/Users/andrew/Downloads/ml_stuttering/clips/stuttering-clips/clips/"
        p = f"C:/Users/andrew/Downloads/ml_stuttering/clips/stuttering-clips/clips/"
        try:
            y, sr = sf.read(p + path_name)
            count += 1
        except Exception as e:
            i+=1
            # print(e)
            continue
        # spectro = mel_spectrogram(y,sr) #128,1100
        mfcc = mfcc_converter(y,sr,n_fft=128,hop_length=1100)
        
        if mfcc.shape != (13,44) or sr <= 0:
            continue
        # if spectro.shape != SPECSHAPE:
        #     print(spectro.shape, "Not proper shape")
        #     i+=1
        #     continue
        # else:
        if row["Prolongation"] > 0 or row["WordRep"] > 0 or row["SoundRep"] > 0 or row["Block"] > 0 or row["Interjection"] > 0: # means they stuttered #row["NoStutteredWords"] >= 2:
            if row["Prolongation"] > 0:
                data["Prolongation"].append(1)
                data["Interjection"].append(0)
                data["Block"].append(0)
                data["SoundRep"].append(0)
                data["WordRep"].append(0)
                data["total"].append(1)
            elif row["WordRep"] > 0:
                data["WordRep"].append(1)
                data["Interjection"].append(0)
                data["Block"].append(0)
                data["SoundRep"].append(0)
                data["Prolongation"].append(0)
                data["total"].append(2)
            elif row["SoundRep"] > 0:
                data["SoundRep"].append(1)
                data["Interjection"].append(0)
                data["Block"].append(0)
                data["WordRep"].append(0)
                data["Prolongation"].append(0)
                data["total"].append(3)
            elif row["Block"] > 0:
                data["Block"].append(1)
                data["Interjection"].append(0)
                data["SoundRep"].append(0)
                data["WordRep"].append(0)
                data["Prolongation"].append(0)
                data["total"].append(4)
            else:
                data["Interjection"].append(1)
                data["Block"].append(0)
                data["SoundRep"].append(0)
                data["WordRep"].append(0)
                data["Prolongation"].append(0)
                data["total"].append(5)

            data["Stutter"].append(1) # did stutter
        else:
            data["Stutter"].append(0) # didnt stutter
            data["Interjection"].append(0)
            data["Block"].append(0)
            data["SoundRep"].append(0)
            data["WordRep"].append(0)
            data["Prolongation"].append(0)
            data["total"].append(0)

        if row["NaturalPause"] > 0:
            data["natural"].append(1)
        else:
            data["natural"].append(0)

        # data["spect"].append(spectro.tolist())
        data["mfcc"].append(mfcc.tolist())

        # print(count, test_c)
        # nn_df = pd.concat([new_df, row], ignore_index=True)

        # new_df = nn_df  # Append the new row to the new DataFrame
        # retval_df = new_df
        # new_df = new_df.append(row, ignore_index=True)  # Append the new row to the new DataFrame TRY CHANGEING IT TO pd.concat
        # retval_df = new_df

    # print(f'Number of times failed: {i}')
    if save:
        with open(JSON_PATH, "w") as fp:
            # print(data)
            json.dump(data, fp, indent=4)
    return data

def print_shape(csv, nftt=512, hoplen=2048):
    count = 0
    i = 0
    for index, row in csv.iterrows():
    # find each wav file
        
        path_name = f'{row["Show"]}_{row["EpId"]}_{row["ClipId"]}.wav'
        p = f"C:/Users/andrew/Downloads/ml_stuttering/clips/stuttering-clips/clips/"
        try:
            y, sr = sf.read(p + path_name)
            count += 1
        except Exception as e:
            i+=1
            # print(e)
            continue
        mfcc = mfcc_converter(y,sr,n_fft=nftt,hop_length=hoplen).reshape(13,44)
        # print(mfcc.shape)
        # mfcc.reshape(13,44)
        print(mfcc.shape)
        if count > 10:
            break

def process(JSONPATH, save=False):

    CSV = "C:/Users/andrew/Downloads/ml_stuttering/fluencybank_labels.csv"
    SEP = "C:/Users/andrew/Downloads/ml_stuttering/SEP-28k_labels.csv"

    csv_file = CSV #create dataframe for testing file
    sep_file = SEP
    csv_df = pd.read_csv(csv_file, delimiter=",", encoding='utf-8')
    sep_df = pd.read_csv(sep_file, delimiter=",", encoding='utf-8')

    new_df = pd.concat([csv_df, sep_df], ignore_index=True)

    csv_df = new_df

    csv_df = csv_df.sample(frac=1)

    csv_df.reset_index(drop=True, inplace=True)
    # print(csv_df.shape)
    data = file_to_df(csv_df,JSONPATH,save)
    # print_shape(csv_df)
    print("Done")
    return


DATA = "C:/Users/andrew/Downloads/total_data.json"
SMALLER = "C:/Users/andrew/Desktop/stutter_model"
def load_data(pathname): #Stutter,Prolongation,WordRep,SoundRep,Block,Interjection
  with open(pathname, "r") as fp:
    data = json.load(fp)
  
  return data

# data = {
#     "Prolongation": [],
#     "WordRep": [],
#     "SoundRep": [],
#     "Block": [],
#     "Interjection": [],
#     "Stutter": [],
#     "spect": [],
#     "mfcc": ["mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12","mfcc_13","mfcc_14"],  # Example "mfcc" list
#     "natural": [],
#     "total": [0, 1, 2, 3, 4, 5, 3, 4, 5,2,1,0, 1, 2]  # Example "total" list
# }

def getType(total_pairs, key, value):
    target_type_pairs = [mfcc for total, mfcc in total_pairs if total == int(key)]
    sample = random.sample(target_type_pairs,value)
    
    return sample


def loadSmallData(DATA):
    data = load_data(DATA)
    
    stut, prol, word, sound, block, interj = data["total"].count(0), data["total"].count(1), data["total"].count(2), data["total"].count(3), data["total"].count(4), data["total"].count(5)
# Binary classification model of based on all the data (23321 instances of stutters and 6330 instances of no stutters)
# Randomly select 6330 instances of stutters (each stutter class is proportional, for example, for 8832 prolongations, we randomly select (8832/23321)*6330 = 2397 samples), then with the 6330 instances of no stutters to construct another binary classification model.
    total = prol + word + sound + block + interj
    balanced = {"1":math.floor(prol*stut/total), "2":math.floor(word*stut/total), "3":math.floor(sound*stut/total), "4":math.floor(block*stut/total), "5":math.floor(interj*stut/total)}
    # print(balanced)
    total_pairs = list(zip(data["total"],data["mfcc"]))
    
    
    mfcc = []
    stuts = 0
    for (k,v) in balanced.items():
        s = getType(total_pairs, k, v)
        for vals in s:
            mfcc.append(vals)
            stuts += 1

    ns = getType(total_pairs, "0", stut)
    for vals in ns:
        mfcc.append(vals)
    result = []
    result.extend([1]*stuts)
    result.extend([0]*stut)
    pairs = list(zip(mfcc, result))
    random.shuffle(pairs)
    mfcc, result = zip(*pairs)

    return {"mfcc":list(mfcc), "Stutter":list(result)}

def load_already_data(data, JSON_PATH="C:/Users/andrew/Downloads/smallData.json"):
     
    with open(JSON_PATH, "w") as fp:
        # print(data)
        json.dump(data, fp, indent=4)

def split_large_json(original_file_path=DATA, output_folder=SMALLER, chunk_size=1000):
    with open(original_file_path, 'rb') as f:
        # Create an ijson parser
        parser = ijson.items(f, 'item')

        current_chunk = 0
        records = []

        for item in parser:
            records.append(item)

            if len(records) == chunk_size:
                output_file_path = f"{output_folder}/part_{current_chunk + 1}.json"
                with open(output_file_path, 'w') as out_file:
                    json.dump({'records': records}, out_file, indent=2)
                
                current_chunk += 1
                records = []

        # Write the last chunk if any remaining records
        if records:
            output_file_path = f"{output_folder}/part_{current_chunk + 1}.json"
            with open(output_file_path, 'w') as out_file:
                json.dump({'records': records}, out_file, indent=2)

def extract_prol(data):
    # print(np.array(data["mfcc"]).shape)
    prol = 500 #data["total"].count(1)
    noStutter = prol
    retval = [[],[]]
    while prol > 0 and noStutter > 0:
        for p,m in zip(data["Prolongation"], data["mfcc"]):
            if p == 0 and noStutter > 0:
                retval[0].append(p)
                retval[1].append(m)
                noStutter -= 1
            elif p == 1 and prol > 0:
                retval[0].append(p)
                retval[1].append(m)
                prol -= 1
                
    return retval 

# data = {"total":[1,0],"mfcc":[[-601.9928369924224, -799.6587101846902, -711.5837809652462, -678.0840118972325, -695.8330450216201, -789.7173421701492, -811.8135627602865, -823.3579646207633, -827.002693193787, -778.9436278259819, -764.1109498432247, -767.4102350675078, -725.3256929358463, -735.7687774596067, -771.0915328757014, -735.7027844806943, -733.2877416540382, -598.4132831925543, -508.0990985402041, -580.2980028261152, -583.0083603142269, -543.3514708527252, -519.0048829823513, -625.1936093205787, -561.4748962169086, -587.4962031438769, -716.5403857736351, -721.3244891198506, -644.9400019034755, -786.6564397183548, -558.5799085349054, -651.5031961494317, -677.6470777431585, -689.0532159369667, -628.2188459527575, -598.2628348261268, -619.6332898318444, -564.9509798204174, -521.3778404585955, -737.7300587522094, -533.9868512262351, -580.6371062677212, -730.751232987047, -606.6495223263408], [-24.50147077755464, 57.96674216291774, 115.68128848021465, 88.60589800079691, 38.87926926899907, 19.802543312723394, 25.06533488498961, 8.627066490392192, 10.169208046871677, 10.839639874962758, 5.687555605117694, 4.828316941634091, -12.722717577444646, 7.538446991095242, 10.824538572759955, 8.750647161567674, -28.031796199034268, 11.959908313486444, -22.299659572722426, -45.375170253975945, -120.23207761789406, -29.181781516572933, -34.763053138075094, -0.3703405974882994, -130.31970310830312, -101.35546818344633, -14.113130501711545, 6.259153468907744, -60.861162622460014, 12.103647872152525, -91.90076939635924, 57.18931709096904, 53.23392121228659, 50.84861829473846, 18.14316689495879, 33.47014991559434, -16.989858492839495, -93.03428741277924, -44.931357847067275, 32.354612910493906, -45.193573904648176, -5.848888414255081, 33.320021996083064, -35.32720770258278]], "Prolongation":[0,1]}

# print(extract_prol(data))

def extract_soundRep(data):
    prol = 500 # data["total"].count(3)
    noStutter = prol
    retval = [[],[]]
    while prol > 0 and noStutter > 0:
        for p,m in zip(data["SoundRep"], data["mfcc"]):
            if p == 0 and noStutter > 0:
                retval[0].append(p)
                retval[1].append(m)
                noStutter -= 1
            elif p == 1 and prol > 0:
                retval[0].append(p)
                retval[1].append(m)
                prol -= 1
                
    return retval  

def extract_wordRep(data):
    prol = 1000 #data["total"].count(2)
    noStutter = prol
    retval = [[],[]]
    while prol > 0 and noStutter > 0:
        for p,m in zip(data["WordRep"], data["mfcc"]):
            if p == 0 and noStutter > 0:
                retval[0].append(p)
                retval[1].append(m)
                noStutter -= 1
            elif p == 1 and prol > 0:
                retval[0].append(p)
                retval[1].append(m)
                prol -= 1
    
    r = {"mfcc":retval[1], "WordRep":retval[0]}
    return r 

# Example usage
# split_large_json(DATA, "C:/Users/andrew/Desktop/stutter_model", chunk_size=1000)

def testNewClips(model):
    norepList = []
    actualNoRep = []
    repList = []
    actualRep = []
    for audio in os.listdir("C:/Users/andrew/Desktop/stutter_model/norepclips"):
        try:
            y, sr = sf.read("norepclips/" + audio)
            mfcc = mfcc_converter(y,sr)
            if mfcc.shape == (94, 13):
                # print("MFCC", mfcc.shape)
                inputs = np.array(mfcc)
                inputs = np.array(inputs.reshape(1, inputs.shape[0], inputs.shape[1], 1))
                # inputs = np.array(inputs.reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2], 1))
                predicted = model.predict(inputs)
                predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
                norepList.append(predicted[0])
                actualNoRep.append(0)
                print("Predicted NoRep: ",predicted)
        except Exception as e:
            print(e)
            break
        

    for audio in os.listdir("C:/Users/andrew/Desktop/stutter_model/repclips"):
        try:
            y, sr = sf.read("repclips/" + audio)
            mfcc = mfcc_converter(y,sr)
            if mfcc.shape == (94, 13):
                # print("MFCC", mfcc.shape)
                inputs = np.array(mfcc)
                inputs = np.array(inputs.reshape(1, inputs.shape[0], inputs.shape[1], 1))
                # inputs = np.array(inputs.reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2], 1))
                predicted = model.predict(inputs)
                predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
                repList.append(predicted[0])
                actualRep.append(1)
                print("Predicted Rep: ",predicted, audio)
        except Exception as e:
            print(e)
            continue
    print("NoRepList", norepList)
    print("RepList",repList)
    return repList, norepList, actualRep, actualNoRep
