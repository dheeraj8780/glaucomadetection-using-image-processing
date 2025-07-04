
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pickle
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageTk 
import cv2
from skimage import color
from skimage.feature import greycomatrix, greycoprops
import scipy.stats as stats
from sklearn import svm
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
from xgboost import XGBClassifier
from keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


global filename
global X,Y
accuracy = []
precision = []
recall = []
fscore = []
global X_train, X_test, y_train, y_test
global cnn
global labels
train_metrics=[]
test_metrics=[]

labels = ['glaucoma','normal']

with open('model/deep_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    cnn_classifier = model_from_json(loaded_model_json)
json_file.close()    
cnn_classifier.load_weights("model/deep_model_weights.h5")
cnn_classifier._make_predict_function() 


main = tkinter.Tk()
main.title("Glaucoma Detection") #designing main screen
main.geometry("1300x1200")

def remove_green_pixels(image):
  # Transform from (256,256,3) to (3,256,256)
  channels_first = channels_first_transform(image)

  r_channel = channels_first[0]
  g_channel = channels_first[1]
  b_channel = channels_first[2]

  # Set those pixels where green value is larger than both blue and red to 0
  mask = False == np.multiply(g_channel > r_channel, g_channel > b_channel)
  channels_first = np.multiply(channels_first, mask)

  # Transfrom from (3,256,256) back to (256,256,3)
  image = channels_first.transpose(1, 2, 0)
  return image

def rgb2lab(image):
  return color.rgb2lab(image)

def rgb2gray(image):
  return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

#feature extraction
def glcm(image, offsets=[1], angles=[0], squeeze=False): #extract glcm features
  single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
  gclm = greycomatrix(single_channel_image, offsets, angles)
  return np.squeeze(gclm) if squeeze else gclm

def histogram_features_bucket_count(image): #texture features will be extracted using histogram
  image = channels_first_transform(image).reshape(3,-1)

  r_channel = image[0]
  g_channel = image[1]
  b_channel = image[2]

  r_hist = np.histogram(r_channel, bins = 26, range=(0,255))[0]
  g_hist = np.histogram(g_channel, bins = 26, range=(0,255))[0]
  b_hist = np.histogram(b_channel, bins = 26, range=(0,255))[0]

  return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
  color_histogram = np.histogram(image.flatten(), bins = 255, range=(0,255))[0]
  return np.array([
    np.mean(color_histogram),
    np.std(color_histogram),
    stats.entropy(color_histogram),
    stats.kurtosis(color_histogram),
    stats.skew(color_histogram),
    np.sqrt(np.mean(np.square(color_histogram)))
  ])

def texture_features(full_image, offsets=[1], angles=[0], remove_green = True):
  image = remove_green_pixels(full_image) if remove_green else full_image
  gray_image = rgb2gray(image)
  glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
  return glcm_features(glcmatrix)

def glcm_features(glcm):
  return np.array([
    greycoprops(glcm, 'correlation'),
    greycoprops(glcm, 'contrast'),
    greycoprops(glcm, 'energy'),
    greycoprops(glcm, 'homogeneity'),
    greycoprops(glcm, 'dissimilarity'),
  ]).flatten()

def channels_first_transform(image):
  return image.transpose((2,0,1))

def extract_features(image):
  offsets=[1,3,10,20]
  angles=[0, np.pi/4, np.pi/2]
  channels_first = channels_first_transform(image)
  return np.concatenate((
      texture_features(image, offsets=offsets, angles=angles),
      texture_features(image, offsets=offsets, angles=angles, remove_green=False),
      histogram_features_bucket_count(image),
      histogram_features(channels_first[0]),
      histogram_features(channels_first[1]),
      histogram_features(channels_first[2]),
      ))

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index 

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')    
    text.insert(END,"Different Labels Found in Dataset : "+str(labels)+"\n\n") 
    text.insert(END,"Total labels in the dataset are : "+str(len(labels)))


    
def featuresExtraction():
    global filename
    global X,Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.npy"):
        X = np.load('model/X.npy')
        Y = np.load('model/Y.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (64,64))
                    class_label = getID(name)
                    features = extract_features(img)
                    Y.append(class_label)
                    X.append(features)
                    print(name+" "+root+"/"+directory[j]+" "+str(features.shape)+" "+str(class_label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X",X)
        np.save("model/Y",Y)
    X = X.astype('float32')
    X = X/255 #features normalization
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Extracted Features : "+str(X[0])+"\n\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset train & test split. 80% dataset images used for training and 20% for testing\n\n")
    text.insert(END,"80% training images : "+str(X_train.shape[0])+"\n\n")
    text.insert(END,"20% testing images : "+str(X_test.shape[0])+"\n\n")

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()

def runSVM():
    global X_train, X_test, y_train, y_test, X, Y
    global accuracy, precision,recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)

    if os.path.exists('model/svm.txt'):
        with open('model/svm.txt', 'rb') as file:
            svm_cls = pickle.load(file)
        file.close()        
    else:
        svm_cls = svm.SVC()
        svm_cls.fit(X, Y)
        with open('model/svm.txt', 'wb') as file:
            pickle.dump(svm_cls, file)
        file.close()
    predict = svm_cls.predict(X_test)
    predict2 = svm_cls.predict(X_train)
    calculateMetrics("SVM", predict, y_test)



def runDeepCNN():
    global X_train, X_test, y_train, y_test, X, Y, cnn
    global accuracy, precision, recall, fscore
    global train_metrics, test_metrics


    # Convert labels to categorical
    Y1 = to_categorical(Y)

    # Reshape for CNN
    XX = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))

    # Split data with stratify
    X_train, X_test, y_train, y_test, Y_train_raw, Y_test_raw = train_test_split(
        XX, Y1, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # Load or build CNN
    if os.path.exists('model/deep_model.json'):
        with open('model/deep_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn = model_from_json(loaded_model_json)
        cnn.load_weights("model/deep_model_weights.h5")
    else:
        cnn = Sequential()
        cnn.add(Convolution2D(64, (3, 3), padding='same', input_shape=(XX.shape[1], XX.shape[2], XX.shape[3]), activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Flatten())
        cnn.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
        cnn.add(Dropout(0.6))
        cnn.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
        cnn.add(Dropout(0.6))
        cnn.add(Dense(Y1.shape[1], activation='softmax'))
        cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = cnn.fit(XX, Y1, batch_size=32, epochs=50, shuffle=True, verbose=2, validation_split=0.1)
        cnn.save_weights('model/deep_model_weights.h5')
        model_json = cnn.to_json()
        with open("model/deep_model.json", "w") as json_file:
            json_file.write(model_json)
        with open('model/deep_history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

    print(cnn.summary())

    # Extract features
    feature_extractor = Model(inputs=cnn.input, outputs=cnn.layers[-3].output)
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # Flatten
    X_train_flat = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test_flat = X_test_features.reshape(X_test_features.shape[0], -1)

    # Class labels
    y_train_labels = np.array(Y_train_raw).astype(int)
    y_test_labels = np.array(Y_test_raw).astype(int)

    num_classes = len(np.unique(y_train_labels))
    if num_classes < 2:
        raise ValueError("Training labels must contain at least two classes.")

    # XGBoost model
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        num_class=num_classes,
        use_label_encoder=False,
        objective='multi:softmax',
        eval_metric='mlogloss'
    )

    xgb.fit(
        X_train_flat, y_train_labels,
        eval_set=[(X_test_flat, y_test_labels)],
        early_stopping_rounds=10,
        verbose=False
    )

    # Predict & Evaluate
    predict_train = xgb.predict(X_train_flat)
    predict_test = xgb.predict(X_test_flat)


    test_metrics = calculateMetrics("Test CNN+XGBoost", predict_test, y_test_labels)




def graph():
    df = pd.DataFrame([
        ['SVM', 'Accuracy', accuracy[0]],
        ['SVM', 'Precision', precision[0]],
        ['SVM', 'Recall', recall[0]],
        ['SVM', 'FScore', fscore[0]],
        ['CNN+XGBoost', 'Accuracy', accuracy[1]],
        ['CNN+XGBoost', 'Precision', precision[1]],
        ['CNN+XGBoost', 'Recall', recall[1]],
        ['CNN+XGBoost', 'FScore', fscore[1]],
    ], columns=['Algorithms','Parameters', 'Value'])

    ax = df.pivot("Algorithms","Parameters", "Value").plot(kind='bar')
    plt.ylim(0,100)  # Set y-axis limits (example: 0 to 100)
    plt.xlim(-0.3,1.8)# Set x-axis limits (adjust as needed)
    plt.xticks(rotation=0) 
    plt.show()



def predict():
    global cnn
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    test = []
    img = cv2.resize(img, (64,64))
    features = extract_features(img)
    test.append(features)
    test = np.asarray(test)
    test = test.astype('float32')
    test = test/255
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    predict = cnn.predict(test)
    predict = np.argmax(predict)

    disease_label = labels[predict]

    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    cv2.putText(img, f'Image Predicted as: {disease_label}', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow(f'Image Predicted as: {disease_label}', img)
    cv2.waitKey(0)

    


def show_both_graphs():
    import json
    import matplotlib.pyplot as plt

    labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']

    # Load CNN + XGBoost metrics
    with open('model/cnn_xgboost_metrics.json', "r") as f:
        data1 = json.load(f)
    train1 = data1.get("Training CNN+XGBoost", [])[:4]
    test1 = data1.get("Test CNN+XGBoost", [])[:4]
    train1 += [0] * (4 - len(train1))
    test1 += [0] * (4 - len(test1))

    # Load CNN + RandomForest metrics
    with open('model/cnn_random_forest.json', "r") as f:
        data2 = json.load(f)
    train2 = data2.get("Training CNN+RandomForest", [])[:4]
    test2 = data2.get("Test CNN+RandomForest", [])[:4]
    train2 += [0] * (4 - len(train2))
    test2 += [0] * (4 - len(test2))

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CNN + XGBoost subplot
    axes[0].plot(labels, train1, marker='o', linestyle='-', color='skyblue', label='Training')
    axes[0].plot(labels, test1, marker='o', linestyle='-', color='salmon', label='Test')
    axes[0].set_title("CNN + XGBoost Performance")
    axes[0].set_ylim(60, 100)
    axes[0].set_xlabel("Metrics")
    axes[0].set_ylabel("Scores")
    axes[0].grid(True)
    axes[0].legend()

    # CNN + RandomForest subplot
    axes[1].plot(labels, train2, marker='o', linestyle='-', color='skyblue', label='Training')
    axes[1].plot(labels, test2, marker='o', linestyle='-', color='salmon', label='Test')
    axes[1].set_title("CNN + RandomForest Performance")
    axes[1].set_ylim(60, 100)
    axes[1].set_xlabel("Metrics")
    axes[1].set_ylabel("Scores")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()



bg_img = Image.open("image.jpg")  
bg_img = bg_img.resize((1920, 1080), Image.ANTIALIAS)  

bg_img_tk = ImageTk.PhotoImage(bg_img)

bg_label = Label(main, image=bg_img_tk)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

   


font = ('times', 16, 'bold')
title = Label(main, text='Glaucoma Detection and Classification using image processing techniques')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=140)       
title.place(x=0,y=0)

font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=178)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=50,y=500)
uploadButton.config(font=font1)  

featuresButton = Button(main, text="Extract Features", command=featuresExtraction)
featuresButton.place(x=370,y=500)
featuresButton.config(font=font1) 

svmButton = Button(main, text="Run SVM", command=runSVM)
svmButton.place(x=650,y=500)
svmButton.config(font=font1)

cnnButton = Button(main, text="Run CNN+XGBoost", command=runDeepCNN)
cnnButton.place(x=50,y=550)
cnnButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=370,y=550)
graphButton.config(font=font1)

graphButton = Button(main, text="Training & Testing Graphs", command=show_both_graphs)
graphButton.place(x=600, y=550)
graphButton.config(font=font1)


predictButton = Button(main, text="Disease Detection", command=predict)
predictButton.place(x=850,y=550)
predictButton.config(font=font1)


main.config(bg='LightSkyBlue')
main.mainloop()
