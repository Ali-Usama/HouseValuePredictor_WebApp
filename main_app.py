from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

cols = ['housing_median_age', 'total_rooms', 'rooms_per_house',
        'total_bedrooms', 'bedrooms_per_rooms', 'population', 'households',
        'population_per_household', 'median_income', 'ocean_proximity']

model = joblib.load('./models/RandomForest.pkl')
scalarObj = joblib.load('./models/scalar.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def get_data():
    housing_age = request.form['Housing_median_age']
    total_rooms = request.form['total_rooms']
    rooms_per_house = request.form['rooms_per_house']
    total_bedrooms = request.form['total_bedrooms']
    bedrooms_per_house = request.form['bedrooms_per_house']
    population = request.form['population']
    households = request.form['households']
    pop_per_household = request.form['pop_per_household']
    median_income = request.form['median_income']
    ocean_proximity = request.form['ocean_proximity']


    def df_final():
        attr_array = np.array(
            [housing_age,total_rooms, rooms_per_house, total_bedrooms, bedrooms_per_house, population, households, pop_per_household, median_income, ocean_proximity])
        final_df = pd.DataFrame([attr_array], columns=cols)

        final_df['ocean_proximity'] = final_df['ocean_proximity'].apply(lambda x: 3.0 if x == 'NEAR BAY' else (
            0.0 if x == '<1H OCEAN' else (1.0 if x == 'INLAND' else (2.0 if x == 'ISLAND' else 4.0))))

        cat_cols = final_df.select_dtypes(include='object').columns
        final_df[cat_cols] = final_df[cat_cols].astype('float')

        return final_df

    def predict(df):
        # cat_cols = [col for col in df.columns if df[col].dtypes == 'object']
        num_cols = [col for col in df.columns if col != 'ocean_proximity']

        arr1 = df[num_cols].values.reshape(-1, 1)
        scaled_arr = scalarObj.fit_transform(arr1)
        encoded_arr = np.array([df['ocean_proximity'].values])

        X_valid1 = np.concatenate((scaled_arr, encoded_arr))

        preds = model.predict(X_valid1.T)
        return preds[0]

    value = f'The value of this house is ${int(predict(df_final()))}'

    return render_template('index.html', result=value)

data = get_data
print(data)

if __name__ == '__main__':
    app.run(debug=True)