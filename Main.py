from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import ttk
from tkinter import filedialog
from tkinter import END  
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import joblib
from scipy.stats import mode


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.layers import Convolution2D,MaxPooling2D,Conv2D,Flatten, Dense, Dropout, BatchNormalization
from keras.models import model_from_json
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import normalize, LabelEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import  ExtraTreesClassifier, VotingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
import lightgbm as lgb


main = Tk()
main.geometry("1300x1200")

global filename
global X, Y
global model
global accuracy
global accuracy, precision, recall, f1

# Initialize empty lists for features and labelsz
X = []
Y = []

base_model = InceptionV3(weights='imagenet', include_top=False)
model_folder = "model"

def uploadDataset():
    global filename,categories
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")

def MobileNetV2_feature_extraction():
    global X, Y, base_model,categories,filename
    text.delete('1.0', END)

    model_data_path = "model/X.npy"
    model_label_path_GI = "model/Y.npy"

    if os.path.exists(model_data_path) and os.path.exists(model_label_path_GI):
        X = np.load(model_data_path)
        Y = np.load(model_label_path_GI)
    else:
 
        X = []
        Y = []
        data_folder=filename
        for class_label, class_name in enumerate(categories):
            class_folder = os.path.join(data_folder, class_name)
            for img_file in os.listdir(class_folder):
                if img_file.endswith('.png'):
                    img_path = os.path.join(class_folder, img_file)
                    print(img_path)
                    img = image.load_img(img_path, target_size=(331,331))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = base_model.predict(x)
                    features = np.squeeze(features)  # Flatten the features
                    X.append(features)
                    Y.append(class_label)
        # Convert lists to NumPy arrays
        X = np.array(X)
        Y = np.array(Y)

        # Save processed images and labels
        np.save(model_data_path, X)
        np.save(model_label_path_GI, Y)
            
    text.insert(END, "Image Preprocessing Completed\n")
    text.insert(END, "InceptionV3 Feature Extraction completed\n")
    text.insert(END, f"Feature Dimension: {X.shape}\n")


  
def Train_test_spliting():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    
    
    X_downsampled = X
    Y_downsampled = Y
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X_downsampled = X_downsampled[indices]
        Y_downsampled = Y_downsampled[indices]  
    else:
        indices = np.arange(X_downsampled.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X_downsampled = X_downsampled[indices]
        Y_downsampled = Y_downsampled[indices]
        
    
    X_train, X_test, y_train, y_test = train_test_split(X_downsampled, Y_downsampled, test_size=0.2, random_state=42)

    text.insert(END, f"Input Data Train  Size: {X_train.shape}\n")
    text.insert(END, f"Input Data Test  Size: {X_test.shape}\n")
    text.insert(END, f"Output  Train Size: {y_train.shape}\n")
    text.insert(END, f"Output  Test Size: {y_test.shape}\n")

    
def performance_evaluation(label,model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred,average='weighted')  
    rec = recall_score(y_true, y_pred,average='weighted')  
    f1s = f1_score(y_true, y_pred,average='weighted')  
    report = classification_report(y_true, y_pred, target_names=label)

    text.insert(END, f"{model_name} Accuracy: {accuracy}\n")
    text.insert(END, f"{model_name} Precision: {pre}\n")
    text.insert(END, f"{model_name} Recall: {rec}\n")
    text.insert(END, f"{model_name} F1-score: {f1s}\n")
    text.insert(END, f"{model_name} Classification report\n{report}\n")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    



def Model_LRC():
    global X_train, X_test, y_train, y_test,Model1
    text.delete('1.0', END)
    num_samples_train, height, width, channels = X_train.shape    
    X_train = X_train.reshape(num_samples_train, height * width * channels)
    
    num_samples_train, height, width, channels = X_test.shape    
    X_test = X_test.reshape(num_samples_train, height * width * channels)

    model_filename  = os.path.join(model_folder, "LRC_model.pkl")
    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = LinearRegression()
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)     
    Y_pred= Model1.predict(X_test)
    performance_evaluation(categories,"Existing Linear Regression", y_test, Y_pred)

def Model_NBC():
    global X_train, X_test, y_train, y_test,Model1
    text.delete('1.0', END)
    model_filename  = os.path.join(model_folder, "NBC_model.pkl")
    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = MultinomialNB()
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)     
    Y_pred= Model1.predict(X_test)
    performance_evaluation(categories,"Existing NBC", y_test, Y_pred)    


def Model_Final():
    global X_train, X_test, y_train, y_test,Model1
    text.delete('1.0', END)
    model_filename  = os.path.join(model_folder, "LGBM_model.pkl")
    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = lgb.LGBMClassifier()
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)     
    Y_pred= Model1.predict(X_test)
    performance_evaluation(categories,"Proposed InceptionV3 with LGBM", y_test, Y_pred)    



def predict():
    global  base_model,categories,optimal_model,Model1

    filename = filedialog.askopenfilename(initialdir="testImages")
    img = image.load_img(filename, target_size=(331, 331))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)    
    features = base_model.predict(x)
    preds  = Model1.predict(features.reshape(1, -1))
    
    if isinstance(preds, (list, np.ndarray)):
        preds = int(preds[0])  # Adjust this line based on the structure of your preds array
    else:
        preds = int(preds)
           
    # Display the result on the image
    img = cv2.imread(filename)
    img = cv2.resize(img, (800, 400))
    
    class_label = categories[preds]

    text_to_display = f'Output Classified as: {class_label}'
    cv2.putText(img, text_to_display,  (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(f'Output Classified as: {class_label}',img)
    cv2.waitKey(0)

def close():
    main.destroy()
    #text.delete('1.0', END)
    
font = ('times', 15, 'bold')
title = Label(main, text='Anemia Diagnosis with Transfer Learning Based Feature  Analysis')
title.config(bg='LightBlue1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)

processButton = Button(main, text="InceptionV3 Feature extraction", command=MobileNetV2_feature_extraction)
processButton.place(x=20,y=150)
processButton.config(font=ff)

processButton = Button(main, text="Train Test Splitting", command=Train_test_spliting)
processButton.place(x=20,y=200)
processButton.config(font=ff)


modelButton = Button(main, text="Existing LRC", command=Model_LRC)
modelButton.place(x=20,y=250)
modelButton.config(font=ff)

modelButton = Button(main, text="Existing NBC", command=Model_NBC)
modelButton.place(x=20,y=300)
modelButton.config(font=ff)

modelButton = Button(main, text="Proposed Methodology", command=Model_Final)
modelButton.place(x=20,y=350)
modelButton.config(font=ff)

predictButton = Button(main, text="Upload test image", command=predict)
predictButton.place(x=20,y=400)
predictButton.config(font=ff)


exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=450)
exitButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config(bg='Pink')
main.mainloop()