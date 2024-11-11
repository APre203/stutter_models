import numpy as np
from keras.models import load_model
from preprocess_data import *
from binaryClassificationCNN import * 
model_pro = load_model("C:/Users/andrew/Downloads/best_model_pro.h5")
model_rep = load_model("C:/Users/andrew/Downloads/best_model_rep.h5")

SMALL_DATA = "C:/Users/andrew/Downloads/smallData.json"

totalData = load_data(SMALL_DATA) 

def detect_prolongation(mfcc):
    # print("MFCC", mfcc) 
    s = 0
    actual = []
    for m in mfcc:
        try:
            m = np.array([m[0],m[12]])
            y = model_pro.predict(m.reshape(1,2,44,1), batch_size=1)
            y = np.around(y,decimals=2)
            if y[0][0] > 0.5:
                s += y[0][0]
                actual.append(1) # correct stutter
            else:
                actual.append(0)
        except:
            actual.append(0)
        
    p_sev = s/len(mfcc)*100
    return p_sev, actual

def detect_repetition(mfcc):
    s = 0
    actual = []
    for m in mfcc:
        # print(m)
        m = np.array(m)
        y = model_rep.predict(m.reshape(1,13,44,1), batch_size=1)
        y = np.around(y,decimals=2)
        try:
            if y[0][0] > 0.5:
                s += y[0][0]
                actual.append(1)
            else:
                actual.append(0)
        except:
            actual.append(0)
    r_sev = s/len(mfcc)*100
    return r_sev, actual

# print(detect_prolongation(totalData["mfcc"]))
# print(np.array(totalData["mfcc"]).reshape(np.array(totalData["mfcc"]).shape[0],13,94,1).shape)

# process("C:/Users/andrew/Downloads/13_44.json",True)
data = load_data("C:/Users/andrew/Downloads/13_44.json")
mfcc_prol = extract_prol(data)
mfcc = mfcc_prol[1]
prol = mfcc_prol[0]

mfcc_rep = extract_soundRep(data)
sound_mfcc = mfcc_rep[1]
soundRep = mfcc_rep[0]


mfcc_wordrep = extract_wordRep(data)
word_mfcc = mfcc_wordrep[1] 
wordRep = mfcc_wordrep[0]



p_serv, predicted = detect_prolongation(mfcc)
print("prol",prol)
print("predicted",predicted)

conf_mat = confusion_matrix(prol, predicted)
print(conf_mat)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()
plt.show()

p_serv, predicted = detect_repetition(sound_mfcc)
print(soundRep)
print(predicted)

conf_mat = confusion_matrix(soundRep, predicted)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()
plt.show()

p_serv, predicted = detect_repetition(word_mfcc)
print(wordRep)
print(predicted)

conf_mat = confusion_matrix(wordRep, predicted)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()
plt.show()
