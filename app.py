from flask import Flask,request, url_for, redirect, render_template

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


app = Flask(__name__)

model=pickle.load(open('modeladaboost2.pkl','rb'))

df = pd.read_csv("milknew.csv", )
Target = {k: v for k, v in zip(['high', 'low', 'medium'], list(range(3)))}
for i in range(df.shape[0]):
    df.iloc[i, -1] = Target[df.iloc[i, -1]]
X = np.array(df.iloc[:, 0:-1])
y = np.asarray(df.iloc[:, -1]).astype('int64')
sc = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y)
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


@app.route('/')
def hello_world():
    return render_template("susu.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]

    reshaped=[np.array(int_features)]
    std_data = sc.transform(reshaped)
    print(int_features)
    print(std_data)
    prediction=model.predict(std_data)
    print(prediction)

    if prediction == 2 :
        return render_template('susu.html',pred='Quality of Milk is High')
    elif prediction == 1 :
        return render_template('susu.html',pred='Quality of Milk is Medium')
    else:
        return render_template('susu.html',pred='Quality of Milk is Low')


if __name__ == '__main__':
    app.run(debug=True)
