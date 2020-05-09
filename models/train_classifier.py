import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    Load data from sqlite database:
    
    Args:
        database_filepath: file path for sqlite database
        
    Returns:
        X variables, Y variables, and category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_query('Select * from messages', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize, lemmatize and normalize text
    
    Args:
        text: raw text
        
    Returns:
        Tokenized, lemmatized and normalized text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Creates sklearn pipeline object:
    
    Args:
        None
        
    Returns:
        Sklearn pipeline object
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [2,5,8],
        'clf__estimator__min_samples_split': [2,3,4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Print performance of model for all categories trying to predict:
    
    Args:
        model: Model to be used for making prediction
        X_test: X variable of test set
        Y_test: Y variable of test set
        category_names: Names of categories
        
    Returns:
        None
    '''
    Y_pred = model.predict(X_test)
    for i in range(0,len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Save model as pickled file
    
    Args:
        model: model to be saved
        model_filepath: file path for the model to be saved
        
    Return:
        None
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    '''
    This will be run will train_classifier.py is executed
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()