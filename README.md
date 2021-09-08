# House Value Predictor GUI

This project includes a Random Forest Model, training on housing dataset, which is deployed on a WebApp using Flask.

## Project Details:
First of all, a RandomForestRegression model is trained on the housing prices dataset, and saved using joblib. This model is then deployed on a WebApp which is developed using  Flask. Follow the steps below in order to successfully run the WebApp:

```
* First, run the `main_app.py` file by moving into the GitHub Directory, and typing `python main_app.py`

* Copy the localhost address into your web browser, and a `web page` will be loaded, asking for several parameters.

* Select an `ocean_proximity` value, and fill in the rest of the values in the `input boxes`.

* Press the `Submit` button, the predicted value will be shown on the same window.

```

*** This project is under development. Further modifications will be committed to this GitHub repository from time to time ***

## Dataset Description:
The dataset used for this project can be found on this [link](https://github.com/ageron/handson-ml2/tree/master/datasets/housing). Some necessary details are given below:

```
>>> housing.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
longitude             20640 non-null float64
latitude              20640 non-null float64
housing_median_age    20640 non-null float64
total_rooms           20640 non-null float64
total_bedrooms        20433 non-null float64
population            20640 non-null float64
households            20640 non-null float64
median_income         20640 non-null float64
median_house_value    20640 non-null float64
ocean_proximity       20640 non-null object
dtypes: float64(9), object(1)
memory usage: 1.6+ MB

>>> housing["ocean_proximity"].value_counts()
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: ocean_proximity, dtype: int64

>>> housing.describe()
          longitude      latitude  housing_median_age   total_rooms  \
count  16513.000000  16513.000000        16513.000000  16513.000000   
mean    -119.575972     35.639693           28.652335   2622.347605   
std        2.002048      2.138279           12.576306   2138.559393   
min     -124.350000     32.540000            1.000000      6.000000   
25%     -121.800000     33.940000           18.000000   1442.000000   
50%     -118.510000     34.260000           29.000000   2119.000000   
75%     -118.010000     37.720000           37.000000   3141.000000   
max     -114.310000     41.950000           52.000000  39320.000000   

       total_bedrooms    population    households  median_income  
count    16355.000000  16513.000000  16513.000000   16513.000000  
mean       534.885112   1419.525465    496.975050       3.875651  
std        412.716467   1115.715084    375.737945       1.905088  
min          2.000000      3.000000      2.000000       0.499900  
25%        295.000000    784.000000    278.000000       2.566800  
50%        433.000000   1164.000000    408.000000       3.541400  
75%        644.000000   1718.000000    602.000000       4.745000  
max       6210.000000  35682.000000   5358.000000      15.000100
```

## Libraries Used in the Project:
An already trained RandomForestRegression Model is included in the repository, so you won't have to train the Model again. The libraries that you'll require to run this script on your system include: 

```
* Flask
* Joblib
* Numpy 
* Pandas
```

#### In order to install these libraries on linux (ubuntu):
```
$ pip3 install Flask
$ pip3 install joblib
$ pip3 install numpy
$ pip3 install pandas
```
