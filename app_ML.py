import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def main():
    activities=['EDA', 'Visualization','Model', 'About us']
    option=st.sidebar.selectbox('Choose your option',activities)
    if option=='EDA':
        st.subheader('Exploratory data analysis')
        data=st.file_uploader('Upload dataset', type=['csv'])
        st.success('File successfully loaded')
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())
            if st.checkbox('Display shape'):
                st.write(df.shape)
            if st.checkbox('Display columns'):
                st.write(df.columns)
            if st.checkbox('Select multiple columns'):
                selected_columns=st.multiselect('Selected columns', df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            if st.checkbox('Display summary'):
                st.write(df1.describe().T)
            if st.checkbox('Display null values'):
                st.write(df1.isnull().sum())
            if st.checkbox('Display correlation'):
                st.write(df1.corr())
    elif option=='Model':
        st.subheader('Model building')
        data=st.file_uploader('Upload dataset', type=['csv'])
        st.success('File successfully loaded')
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())
            X=df.iloc[:,1:-1]
            y=df.iloc[:,-1:]
            seed=st.sidebar.slider('Seed',1,10)
            classifier=st.sidebar.selectbox('Select your preferred classifier', ('KNN', 'SVM'))
            def add_params(clf):
                params=dict()
                if clf=='SVM':
                   C=st.sidebar.slider('C',0.01,15.0)
                   params['C']=C
                elif clf=='KNN':
                   K=st.sidebar.slider('C',1,15)
                   params['K']=K
                return params

            def add_classifier(nclf,params):
                clf=None
                if nclf=='SVM':
                   clf=SVC(C=params['C'])
                elif nclf=='KNN':
                   clf=KNeighborsClassifier(n_neighbors=params['K'])
                else:
                   st.warning('Select your clasifier')
                return clf
            params=add_params(classifier)
            clf=add_classifier(classifier, params)
            clf.fit(X,y)
            ypr=clf.predict(X)
            acc=accuracy_score(y,ypr)
            st.write('Accuracy:',acc)
if __name__=='__main__':
    main()
