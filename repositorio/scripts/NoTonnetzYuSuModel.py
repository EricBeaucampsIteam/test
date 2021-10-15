### Load necessary libraries ###
import glob
import os
import librosa
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Dropout, Flatten
import tensorflow as tf
from tensorflow import keras
from datetime import datetime




def extract_features(parent_dir,sub_dirs,file_ext="*.wav",
                     bands=60,frames=41):
    # Calculo de la primera y última muestra de la trama
    def _windows(data, window_size):
        start = 0
        while start < len(data):
            yield int(start), int(start + window_size)
            start += (window_size // 2)
    #Tamaño de la ventana
    window_size = 512 * (frames - 1)
    features, labels = [], [] 
    for fn in glob.glob(os.path.join(parent_dir, sub_dirs, file_ext)):
        segment_features,feature_mix, segment_labels = [],[],[]
        sound_clip,sr = librosa.load(fn)
       # Etiquetado #
        label = fn.split('-')
        if len(label) < 4 :
            label = fn.split('_')[1].split('0')[1]
            if int(label) < 283:
                label = '3'
            else:
                label = '4'
        else:
            label = fn.split('-')[2]
            if label == '1':
                label = '0'
            if label == '5':
                label = '1'
            if label == '8':
                label = '2'
        print(fn,label)
        # Fin de etiquetado #
        # Se coge el número de muestras definido por _windows para conseguir fragmentos de 23 ms
        for (start,end) in _windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
            # Extracción de características
                signal = sound_clip[start:end]
                stft = np.abs(librosa.stft(signal))
                chroma =librosa.feature.chroma_stft(S=stft, sr=sr).T
                melspec = librosa.feature.melspectrogram(signal, sr=sr, win_length=512, n_mels = bands)
                logspec = librosa.power_to_db(melspec, ref=np.max).T
                contrast = librosa.feature.spectral_contrast(S=stft, sr=sr).T
                feature_mix = np.hstack((logspec,chroma,contrast))
                # Solo se añaden las tramas que no son ceros (Que no hay silencio)
                if len(chroma) > 0:
                    features.append(feature_mix)
                    labels.append(label)
    # Reestructuración para formato de entrada para CNN
    features = np.asarray(features).reshape(len(features),41,79,1)
    # Se añade array de ceros en el último eje
    features = np.concatenate((features, np.zeros(np.shape(features))), axis = 3)
    for i in range(len(features)):
        # Calculo Delta y copiado en el último eje
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    return features, labels
    
timer = []
start_load = 0
parent_dir = r'C:\Users\gtac\Desktop\SSEnCE-rep\UrbanSound8K\audioFiltered2.0'
save_dir =  r"C:\Users\gtac\Desktop\SSEnCE-rep\UrbanSound8K\processed2.0/"
folds = sub_dirs = np.array(['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10'])
for sub_dir in sub_dirs:
    start_extract = datetime.now()
    print('mean feat time: ',np.mean(timer))
    features, labels = extract_features(parent_dir,sub_dir)
    # Se guardan las caracterisitcas y las etiquetas en ficheros .npz
    np.savez("{0}{1}".format(save_dir, sub_dir),
             features=features,
             labels=labels)

### Define convolutional network architecture ###
def get_network():
    num_filters = [24,32,64,128]
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (41,79,2)
    num_classes = 10
    keras.backend.clear_session()
    #Capa 1. 32 neuronas, Batch normalitation, Función de activación: relu
    model = keras.models.Sequential()   
    model.add(keras.layers.Conv2D(32, kernel_size,
                padding="same", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    #Capa 2. 32 neuronas, Batch normalitation, Max Pooling, Función de activación: relu
    model.add(keras.layers.Conv2D(32, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    #Capa 3. 64 neuronas, Batch normalitation, Max Pooling,  Función de activación: relu
    model.add(keras.layers.Conv2D(64, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    #Capa 4. 64 neuronas, Batch normalitation, Función de activación: relu
    model.add(keras.layers.Conv2D(64, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    #Capa 5. 1024 neuronas, Max Pooling, Función de activación: sigmoid
    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dense(1024, activation="sigmoid"))
    #Capa 6 (output). 10 clases, funcion de activacion: softmax
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    # Hyperparametros. Función de pérdidas: Categorical Crossentrpy, optimización del 0.004
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])
    return model


### Entrenamiento siguiendo la distribución 10-Folds cross-validation ###

folds = np.array(['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10'])
load_dir =  r"C:\Users\gtac\Desktop\SSEnCE-rep\UrbanSound8K\processed2.0"

kf = KFold(n_splits=10)
cont = 0
for train_index, test_index in kf.split(folds):
    x_train, y_train = [], []
    # contador para indicar el número de modelo correspondiente
    cont = cont + 1
    for ind in train_index:
        # Obtener las caracterísitcas y etiquetas para los datos de entrenamiento de los cuales se eligirán
        # todas las carpetas menos una, la cuál irá variando en las 10 iteraciones y que posteriormente
        # se utilizará para testeo
        train_data = np.load("{0}/{1}.npz".format(load_dir,folds[ind]),
                       allow_pickle=True)
        # for training stack all the segments so that they are treated as an example/instance
        features = train_data["features"]
        labels =train_data["labels"]
        x_train.append(features)
        y_train.append(labels)
    # stack x,y pairs of all training folds
    x_train = np.concatenate(x_train, axis = 0).astype(np.float32)
    y_train = np.concatenate(y_train, axis = 0).astype(np.float32)

    # Se obtienen las características y etiquetas de 1 de las carpetas para testeo
    test_data = np.load("{0}/{1}.npz".format(load_dir,
                   folds[test_index][0]), allow_pickle=True)
    x_test = test_data["features"]
    y_test = test_data["labels"]
    #Se carga la estructura de CNN previamente definida
    model = get_network()
    model_name = "no" + str(cont) +"_2LMCNN25.h5"
    #Por último se entrena y guarda el modelo en la ruta deseada
    model.fit(x_train, y_train, epochs = 70, batch_size = 32, verbose = 1)
    model.save(r"C:\Users\gtac\Desktop\SSEnCE-rep\UrbanSound8K\models/" + model_name)


