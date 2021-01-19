import pandas as pd

df = pd.read_csv("car_data_km_1000.csv", sep=';')


def dummies(x, df):
    temp = pd.get_dummies(df[x], drop_first=True)
    df = pd.concat([df, temp], axis=1)
    df.drop([x], axis=1, inplace=True)
    return df


# Applying the function to the cars_lr

df = dummies('CEKIS', df)
df = dummies('Gear', df)
df = dummies('Fuel', df)

from sklearn import preprocessing
import pandas as pd

le = preprocessing.LabelEncoder()

df[['Brand', 'Serie', 'Color']] = df[['Brand', 'Serie', 'Color']].apply(le.fit_transform)
# df['city'] = le.fit(df['city'])

import numpy as np
from flask import Flask, render_template, request
import pickle  # Initialize the flask App


X = df.loc[:, ['Brand', 'Serie', 'Color', 'Year', 'KM', 'CC', 'HP',
               'Galeriden', 'GARANTI',
               'Onden', 'Otomatik', 'Yarı',
               'LPG']]

y = df.loc[:, ['Price']]

from sklearn.model_selection import train_test_split

y_test: object
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train.values.ravel())

'''
y_pred = rf_reg.predict(X_test)
print("Accuracy on Traing set: ",rf_reg.score(X_train,y_train))
print("Accuracy on Testing set: ",rf_reg.score(X_test,y_test))
'''
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.

# import libraries
import numpy as np
from flask import Flask, render_template, request
import pickle  # Initialize the flask App

app = Flask(__name__)
pickle.dump(rf_reg, open('model.pkl','wb'))

# default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')





# To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='CO2    Emission of the vehicle is :{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
'''
try:
    import dill as pickle
except ImportError:
    import pickle

pickle.dump(rf_reg, open('model.pkl','wb'))


#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''
