
#------------------importing the required libraries----------------
import os
import warnings
import sys
import pandas as pd     #datamanipulation   
import numpy as np      #scientific calculations
#sklearn
from sklearn.model_selection import train_test_split        #spliting the data to trine and test
from sklearn.metrics import f1_score                #f1 score
from imblearn.over_sampling import RandomOverSampler            #random sampling
#model selection from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#mlflow
import mlflow
import mlflow.sklearn
#streamlet
import streamlit as st
import pickle

#setting the dataset path
path = "dvc_data\survey lung cancer.csv"
#creating the dataset 
data = pd.read_csv(path)
print(data.shape)           #shape of the data before removing the duplicated
#finding the duplicates
duplicate = data[data.duplicated()]
print(duplicate.index)
#removing the duplicated
for i in duplicate.index:
        print( "index ", i, " is removed and no longer available")
        data.drop(index=[i], inplace = True)
        data.reset_index()
print(data.shape)       #shape of the data after removing the duplicates
#changing the catogorical values of features GENDER and LUNG_CANCER
data.replace({'GENDER':{'F':0,'M':1}},inplace=True)
data.replace({'LUNG_CANCER':{'YES':0,'NO':1}},inplace=True)
#split the data inti traing and test set (75% and 25%)
X = data.drop(["LUNG_CANCER"],axis=1)      #required variable
y = data.LUNG_CANCER            #dependent variable
#oversampling the data
over_samp =  RandomOverSampler(random_state=0)
X_train_res, y_train_res = over_samp.fit_resample(X, y)
#spliting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_train_res, y_train_res,test_size= 0.2, random_state = 0)
#selecting the best model
def select_best_model(maximum, dc, rf, knn ,svc):
    if maximum == dc:
        a = dc
        b = DC
        predict = DC.predict(X_test)
        f1_score_dc = f1_score(y_test, predict)
        print(f1_score_dc)
        c = f1_score_dc
        #print("DC")
        return(a,b,c)
    elif maximum == rf:
        #print("RF")
        a = rf
        b = RF
        predict = DC.predict(X_test)
        f1_score_rf = f1_score(y_test, predict)
        print(f1_score_rf)
        c = f1_score_rf
        return(a,b,c)
    elif maximum == knn:
        #print("KNN")
        a = knn
        b = KNN
        predict = DC.predict(X_test)
        f1_score_knn = f1_score(y_test, predict)
        print(f1_score_knn)
        c = f1_score_knn
        return(a,b,c)
    else:
        #print(SVC)
        a = svc
        b = SVC
        predict = DC.predict(X_test)
        f1_score_svc = f1_score(y_test, predict)
        print(f1_score_svc)
        c = f1_score_svc
        return(a,b,c)
#running the mlflow
with mlflow.start_run():
    DC = DecisionTreeClassifier()
    RF = RandomForestClassifier()
    KNN = KNeighborsClassifier()
    SVC = SVC()
    accuracy=[]     #create a list to append the accuracy of all models
    for i in range(1):
        DC.fit(X_train, y_train)  #dacision tree classifier nmodel
        dc = round(DC.score(X_test, y_test)*100)        #roundfigure of the accuracy
        accuracy.append(dc)  
        #Random forest    
        RF.fit(X_train, y_train)
        print("Accuracy obtained by RandomForestClassifier model: " + str(RF.score(X_test, y_test)*100) + " %")
        rf = round(RF.score(X_test, y_test)*100)
        accuracy.append(rf)
        #KNN
        KNN.fit(X_train, y_train)
        print("Accuracy obtained by RandomForestClassifier model: " + str(KNN.score(X_test, y_test)*100) + " %")
        knn = round(KNN.score(X_test, y_test)*100)
        accuracy.append(knn)
        #SVC
        SVC.fit(X_train, y_train)
        print("Accuracy obtained by RandomForestClassifier model: " + str(SVC.score(X_test, y_test)*100) + " %")
        svc = round(SVC.score(X_test, y_test)*100)
        accuracy.append(svc)
    maximum = max(accuracy)
    print("-----------BEST OF ALL MODEL-----------")
    print(select_best_model(maximum, dc, rf, knn ,svc))
    a,b,c = select_best_model(maximum, dc, rf, knn ,svc)         #assigning the vvalues to a,b,c
    #saving to the pickle file after selecting the best model
    import pickle
    pickle_out = open("pickle_file.pkl", "wb")
    pickle.dump(b,pickle_out)
    pickle_out.close()
    print("Pickle file is ready")

    #mlflow metrics
    mlflow.log_metric("fl_value",c)
    #log parameters
    mlflow.log_param("KNN", knn)
    mlflow.log_param("DC", dc)
    mlflow.log_param("RF", rf)
    mlflow.log_param("SVC", svc)
    #
    mlflow.sklearn.log_model(KNN, "knn")
    mlflow.sklearn.log_model(DC, "dc")
    mlflow.sklearn.log_model(RF, "rf")
    mlflow.sklearn.log_model(SVC, "svc")

#sltreamlit ui
pickle_in = open("pickle_file.pkl", "rb")
classifier = pickle.load(pickle_in)
#creating the container
header = st.container()
header_two = st.container()
def prediction(gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol, coughing, shortness_of_breath, swallowing_difficulty, chest_pain):
    predict = classifier.predict([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])
    return predict

with header:
    st.title("LUNG CANCER PREDICTIONS")
    st.text("The effictuveness of cancer prediction system helps the people to ")
    st.text("know their cancer risk with low cost and it also helps the people to take  ")
    st.text("and it also helps the people to take the appropriate decision based on their ")
    st.text("cancer risk status.")

with header_two:
    st.header("Predict your risk from Lung cancer")
    gender = st.text_input("Gender (Female = 0, Male = 1)")
    age = st.text_input("Age (Enter your age)")
    smoking = st.text_input("Smoking (Yes = 2, No = 1)")
    yellow_fingers = st.text_input("Yellow Fingers (Yes = 2, No = 1)")
    anxiety = st.text_input("Anxiety (Yes = 2, No = 1)")
    peer_pressure = st.text_input("Peer Pressure (Yes = 2, No = 1)")
    chronic_disease = st.text_input("Chronic Disease  (Yes = 2, No = 1)")
    fatigue = st.text_input("Fatigue (Yes = 2, No = 1)")
    allergy = st.text_input("Allergy (Yes = 2, No = 1)")
    wheezing = st.text_input("Wheezing (Yes = 2, No = 1)")
    alcohol = st.text_input("Alcohol (Yes = 2, No = 1)")
    coughing = st.text_input("Coughing (Yes = 2, No = 1)")
    shortness_of_breath = st.text_input("Shortness of Breath (Yes = 2, No = 1)")
    swallowing_difficulty = st.text_input("Swallowing Difficulty (Yes = 2, No = 1)")
    chest_pain = st.text_input("Chest Pain (Yes = 2, No = 1)")
    result = ''
    if st.button("Predict"):
        result = prediction(gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol, coughing, shortness_of_breath, swallowing_difficulty, chest_pain)
        if result == 1:
            st.text("there is a chance to get CANCER" + str(result))
        else:
            st.text("No CANCER"+ str(result))