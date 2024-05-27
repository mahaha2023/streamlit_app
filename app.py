import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay,PrecisionRecallDisplay,precision_recall_curve,roc_curve


def main():
# Your streamlit code
   
    st.title("Application de Machine Learning pour la Détection de Fraude par Carte de Crédit")
    st.subheader("Auteur : Mahamat Hassan Issa")
    if st.sidebar.checkbox("Description de données",False):
        st.write("Les ensembles de données contiennent les transactions effectuées par carte de crédit en septembre 2013 par des titulaires de cartes européens. Cet ensemble de données présente les transactions survenues en deux jours, où nous avons 492 fraudes sur 284 807 transactions.")
        st.write("L'ensemble de données est très déséquilibré, la classe positive (fraudes) représentant 0,172% de toutes les transactions.  Il contient uniquement des variables d'entrée numériques qui sont le résultat d'une transformation PCA.")
        st.write("En raison de problèmes de confidentialité, les caractéristiques originales et davantage d'informations générales sur les données ne sont pas fournies.Les fonctionnalités V1, V2, ... V28 sont les principaux composants obtenus avec PCA; Les fonctionnalités V1, V2, ... V28 sont les principaux composants obtenus avec PCA.")
        st.write("Les seules fonctionnalités qui n'ont pas été transformées avec PCA sont Time et Amount. Feature Time contient les secondes écoulées entre chaque transaction et la première transaction de l'ensemble de données. La fonctionnalité Montant est le montant de la transaction, cette fonctionnalité peut être utilisée pour un apprentissage sensible aux coûts en fonction d'un exemple.")

                        



    #Fonction d'importation de données
    
    # Affichage de la table de données
    @st.cache(persist=True)   
    def load_data():
      data = pd.read_csv("creditcard.csv")
      return data
    
    df = load_data()
    
    if st.sidebar.checkbox("Afficher les données brutes",False):
       st.subheader("Données")
    
       
       st.write(df)
    
    # trai/test split
    
    y=df['Class']
    x=df.drop('Class',axis=1)
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2,random_state=42,stratify=y)
    st.write("X_train: ",x_train.shape)
    st.write("X_test: ",x_test.shape)
    st.write("y_train :",y_train.shape)
    st.write("y_test :",y_test.shape)
    
    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)



        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix") 
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot()
            st.pyplot()
        
        

           
            

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            disp.plot()
            st.pyplot()

   
   
        
    # Choix des classificateur 
        
    st.sidebar.subheader("Choisir un modèle")
    classifier = st.sidebar.selectbox("modèle", ("Random Forest","Logistic Regression","Support Vector Machine (SVM)"))
            
            
   
            
        
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Optimisation des Hyperparamètres")
        C = st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel",("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient", ("scale", "auto"), key = 'gamma')
        metrics = st.sidebar.multiselect("Choisir une Courbe de Métrique",('Confusion Matrix', 'Precision-Recall Curve'))

        if st.sidebar.button("Exécuter", key='classify'):
            st.subheader("Support Vector Machine (SVM Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=model.classes_))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=model.classes_))
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Optimisation des Hyperparamètres")
        C = st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("max_iter", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("Choisir une Courbe de Métrique",('Confusion Matrix', 'Precision-Recall Curve'))

        if st.sidebar.button("Exécuter", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=model.classes_))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=model.classes_))
            plot_metrics(metrics)


 
    if classifier == 'Random Forest':
        st.sidebar.subheader("Optimisation des Hyperparamètres")
        n_estimators  = st.sidebar.number_input("n_estimators", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("max_depth", 1, 20, step=1, key='max_depth')
        #bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True','False'), key='bootstrap')
        metrics = st.sidebar.multiselect("Choisir une Courbe de Métrique",('Confusion Matrix', 'Precision-Recall Curve'))

        if st.sidebar.button("Exécuter", key='classify'):
            st.subheader("")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=model.classes_))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=model.classes_))
            plot_metrics(metrics)


    
    
  
        
  
