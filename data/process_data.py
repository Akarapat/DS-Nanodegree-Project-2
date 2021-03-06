# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load and merge data from two csv files
    
    Args:
        messages_filepath: file path for disaster_messages.csv
        categories_filepath: file path for disaster_categories.csv
        
    Returns:
        combined dataset
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.join(categories.set_index('id'), on='id')
    return df

def clean_data(df):
    '''
    Clean data
    
    Args:
        df: Pandas dataframe
        
    Returns:
        Cleaned data
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0].tolist()
    category_colnames = [x.split('-')[0] for x in row]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].astype(int)
    df.drop(['categories'], axis=1, inplace=True)
    df = df.join(categories, lsuffix='_caller', rsuffix='_other')
    df = df[~df.duplicated(keep='first')]
    df['related'] = df['related'].replace(2, 1)
    return df

def save_data(df, database_filename):
    '''
    Save data into sqlite database
    
    Args:
        df: Pandas dataframe
        database_filename: File name of database
        
    Returns:
        None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    '''
    Function that will be called when process_data.py is executed
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()