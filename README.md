# Disaster Response Pipeline Project

### Installation:
1.  The script run on Python 3.6.3
2.  Following packages are required to run the program
	- scikit-learn
    - pandas
    - numpy
    - sqlalchemy
    - nltk
    - plotly
    - flask

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project motivation
This project is included in Data Scientist nanodegree of Udacity.  The aim is to help Figure Eight to classify messages according to related disaster.  It is very important to timely identify related disaster so that appropriated responses can be arranged

### Results
After training te model with data from Figure Eight, it is possible to classify messages according to related disaster with reasonable accuracy

### Creator
Akarapat Charoenpanich

### Thanks
Udacity and Figure Eight