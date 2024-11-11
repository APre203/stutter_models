from miscModels.preprocess_data import *
# from LSTM import LSTM
from NewModels.neuralNetwork import *
from miscModels.wordRepModel import *
# from multiClassification import MultiClassification

# from binaryClassification import BinaryClassification
# from binaryClassificationCNN import BinaryClassificationCNN

# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi

def main():
    DATA = "C:/Users/andrew/Downloads/total_data.json"
    DATA_T = "C:/Users/andrew/Downloads/data_t.json"
    SMALL_DATA = "C:/Users/andrew/Downloads/smallData.json" # mfcc and Overall Stutter
    data = load_data(DATA)
    # data = loadSmallData(DATA)
    # data = extract_wordRep(data)
    # model_word = NeuralNetworkModel(data, "WordRep")
    # hist, actual, predicted, model = model_word.train_model()
    # model_word.plot_hist(hist)
    # model_word.plot_auc(actual, predicted)
    
    word_model = WordRepModel(data, "WordRep")
    word_model.train_model()

    # model_pkl_file = "word_rep_model.pkl"  

    # # with open(model_pkl_file, 'wb') as file:  
    # #     pickle.dump(model, file)

    # with open(model_pkl_file, 'rb') as file:  
    #     model = pickle.load(file)
    # rep, norep, actrep, actnorep = testNewClips(model)
    # actual = actrep+actnorep
    # predicted = rep+norep
    # print(actual)
    # print(predicted)
    # conf_mat = confusion_matrix(actual, predicted)
    # displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    # displ.plot()
    # plt.show()
    # evaluate model 
    # y_predict = model.predict(X_test)

    # model_LSTM = LSTM(data)
    # hist, actual, predicted = model_LSTM.train_model(epochs=100, batch_size=128) #120, 4
    # model_LSTM.plot_hist(hist)
    # model_LSTM.plot_auc(actual,predicted)



    # model_binary = BinaryClassification(data)
    # hist, actual, predicted = model_binary.train_model(epochs=100, batch_size=4) #120, 4
    # model_binary.plot_hist(hist)
    # model_binary.plot_auc(actual,predicted)


    
    # model_prol = NeuralNetworkModel(data, "Prolongation")
    # hist = model_prol.train_model()
    # model_prol.plot_hist(hist)

    # model_word = NeuralNetworkModel(data, "WordRep")
    # hist = model_word.train_model()
    # model_word.plot_hist(hist)

    # model_sound = NeuralNetworkModel(data, "SoundRep")
    # hist = model_sound.train_model()
    # model_sound.plot_hist(hist)

    # model_block = NeuralNetworkModel(data, "Block")
    # hist = model_block.train_model()
    # model_block.plot_hist(hist)

    # model_inter = NeuralNetworkModel(data, "Interjection")
    # hist = model_inter.train_model()
    # model_inter.plot_hist(hist)

    # model_natural = NeuralNetworkModel(data, "natural")
    # hist = model_natural.train_model()
    # model_natural.plot_hist(hist)

    # model_multi = MultiClassification(data)
    # hist = model_multi.train_model()
    # model_multi.plot_hist(hist)
    
    print("Done")
    return "Done"


if __name__ == '__main__':
    main()