from sklearn.ensemble import RandomForestRegressor
from datetime import date
import sys
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from treeinterpreter import treeinterpreter as ti
import numpy as np

predpath = r'Predictions\prediction_data'
aggpath = r'AggData\aggdata'
aggdatapath = r'AggData\data'
twotpath = r'Twot\data'

date_time = date.today()


## SET UP ##

# DB Functions
def connect():
    """ Connect to the PostgreSQL database server """
    try:
        conn = psycopg2.connect(
            host='138.68.140.227',
            database="postgres",
            user="postgres",
            password="password",
            port=25432
        )
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1)
    print("Connection successful")
    return conn


def postgresql_to_dataframe(select_query, column_names):
    """
    Tranform a SELECT query into a pandas dataframe
    """
    conn = connect()
    cursor = conn.cursor()
    try:
        cursor.execute(select_query)
        print('')
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        cursor.close()
        return 1

    # Naturally we get a list of tupples
    tupples = cursor.fetchall()
    cursor.close()

    # We just need to turn it into a pandas dataframe
    df = pd.DataFrame(tupples, columns=column_names)

    conn.close()
    return df




connect()

# Get data

column_names = ['id', 'pos', 'neg', 'neu', 'comp', 'brand', 'day', 'ticker', 'closeprice', '5dayMA', 'EMA', 'fut1',
                'fut2', 'fut3', 'fut4', 'fut5']
df = postgresql_to_dataframe('select * from fypapp_aggdata',
                             column_names)

# Cant deal with 0 or nan, nans will be removed
df['5dayMA'].replace(0, np.nan, inplace=True)
UniqueNames = df.brand.unique()

# Get data

DataFrameDict = {elem: pd.DataFrame for elem in UniqueNames}

for key in DataFrameDict.keys():
    DataFrameDict[key] = df[:][df.brand == key]

df_dict = DataFrameDict

dic = dict()
for key in df_dict:
    if not df_dict[key].empty:
        dic[key] = df_dict[key]

df_dict = dic

# Declarations

scr5day = 0
scr5dayEMA = 0
scrMA = 0
scr = 0

scr5day_list = np.array([])
scr5dayEMA_list = np.array([])
scrMA_list = np.array([])
scr_list = np.array([])
full_predictions = pd.DataFrame()
df_list = list(df_dict)
size = len(df_list)
prediction_list = []
i = 0
df_brands = list()
count = 0
score_list = pd.DataFrame()
## For all Brands
for key in df_list:
    x = df_list[i]
    df_new = df_dict[x]
    # Dont bother with predictions if there is less than 30 days
    if len(df_new) > 30:
        df_brands.append(df_list[i])
        count = count + 1

        predictions = pd.DataFrame()

        df_new = df_new.set_index('day')
        df_new.dropna(inplace=True)

        # Start ML Make fits and predictions for different variations of the Model
        # Purely sentiment, Differernt moving averages and combination models
        # The goal is too have a Sentiment Moving Avg combo more accurate than other possible models

        # Split Data into training and testing
        train, test = train_test_split(df_new, test_size=.35, random_state=20)

        # Fit Model

        rf_sent = RandomForestRegressor(n_estimators=500)
        rf_sent.fit(train[['pos', 'neu', 'neg', 'comp']], train['closeprice'])

        rf_5dayMA = RandomForestRegressor(n_estimators=500)
        rf_5dayMA.fit(train[['5dayMA']], train['closeprice'])

        rf_5dayEMA = RandomForestRegressor(n_estimators=500)
        rf_5dayEMA.fit(train[['5dayMA', 'EMA']], train['closeprice'])

        rf_MA = RandomForestRegressor(n_estimators=500)
        rf_MA.fit(train[['EMA']], train['closeprice'])

        rf_sentMA = RandomForestRegressor(n_estimators=500)
        rf_sentMA.fit(train[['pos', 'neu', 'neg', 'comp', 'EMA', '5dayMA']], train['closeprice'])

        # Make prediction lists - Can be Refactored

        prediction_list_sent = []
        prediction_sent, bias, contributions = ti.predict(rf_sent, test[['pos', 'neu', 'neg', 'comp']])
        prediction_list.append(prediction_sent)
        predictions_dataframe_list_sent = pd.DataFrame(data=prediction_sent[0:], columns=['closeprice'])
        predictions_dataframe_list_sent['closeprice'] = predictions_dataframe_list_sent['closeprice'] + 0
        test.reset_index(inplace=True)
        predictions_dataframe_list_sent['actual_value'] = test['closeprice']
        predictions_dataframe_list_sent['Brand'] = df_new.brand[0]
        predictions_dataframe_list_sent.columns = ['pred_sentiment', 'actual_price', 'Brand']

        prediction_list_5dayMA = []
        prediction_5dayMA, bias, contributions = ti.predict(rf_5dayMA, test[['5dayMA']])
        prediction_list.append(prediction_5dayMA)
        predictions_dataframe_list_5dayMA = pd.DataFrame(data=prediction_5dayMA[0:], columns=['closeprice'])
        predictions_dataframe_list_5dayMA['closeprice'] = predictions_dataframe_list_5dayMA['closeprice'] + 0
        predictions_dataframe_list_5dayMA['actual_value'] = test['closeprice']
        predictions_dataframe_list_5dayMA['Brand'] = df_new.brand[0]
        predictions_dataframe_list_5dayMA.columns = ['pred_5day', 'actual_price', 'Brand']

        prediction_list_5dayEMA = []
        prediction_5dayEMA, bias, contributions = ti.predict(rf_5dayEMA, test[['5dayMA', 'EMA']])
        prediction_list.append(prediction_5dayEMA)
        predictions_dataframe_list_5dayEMA = pd.DataFrame(data=prediction_5dayEMA[0:], columns=['closeprice'])
        predictions_dataframe_list_5dayEMA['closeprice'] = predictions_dataframe_list_5dayEMA['closeprice'] + 0
        predictions_dataframe_list_5dayEMA['actual_value'] = test['closeprice']
        predictions_dataframe_list_5dayEMA['Brand'] = df_new.brand[0]
        predictions_dataframe_list_5dayEMA.columns = ['pred_5dayEMA_combo', 'actual_price', 'Brand']

        prediction_list_MA = []
        prediction_MA, bias, contributions = ti.predict(rf_MA, test[['EMA']])
        prediction_list.append(prediction_MA)
        predictions_dataframe_list_MA = pd.DataFrame(data=prediction_MA[0:], columns=['closeprice'])
        predictions_dataframe_list_MA['closeprice'] = predictions_dataframe_list_MA['closeprice'] + 0
        test.reset_index(inplace=True)
        predictions_dataframe_list_MA['actual_value'] = test['closeprice']
        predictions_dataframe_list_MA['Brand'] = df_new.brand[0]
        predictions_dataframe_list_MA.columns = ['pred_EMA', 'actual_price', 'Brand']

        prediction_list_sentMA = []
        prediction_sentMA, bias, contributions = ti.predict(rf_sentMA,
                                                            test[['pos', 'neu', 'neg', 'comp', 'EMA', '5dayMA']])
        prediction_list.append(prediction_sentMA)
        predictions_dataframe_list_sentMA = pd.DataFrame(data=prediction_sentMA[0:], columns=['closeprice'])
        predictions_dataframe_list_sentMA['closeprice'] = predictions_dataframe_list_sentMA['closeprice'] + 0
        test.reset_index(inplace=True)
        predictions_dataframe_list_sentMA['actual_value'] = test['closeprice']
        predictions_dataframe_list_sentMA['Brand'] = df_new.brand[0]
        predictions_dataframe_list_sentMA.columns = ['pred_sentiment_EMA_combo', 'actual_price', 'Brand']

        # Get values for getting the R2 Scores of each model

        import sklearn.metrics as sm

        y_test_sent = predictions_dataframe_list_sent['actual_price']
        y_test_pred_sent = predictions_dataframe_list_sent['pred_sentiment']

        y_test_MA = predictions_dataframe_list_MA['actual_price']
        y_test_pred_MA = predictions_dataframe_list_MA['pred_EMA']

        y_test_5dayMA = predictions_dataframe_list_5dayMA['actual_price']
        y_test_pred_5dayMA = predictions_dataframe_list_5dayMA['pred_5day']

        y_test_5dayEMA = predictions_dataframe_list_5dayEMA['actual_price']
        y_test_pred_5dayEMA = predictions_dataframe_list_5dayEMA['pred_5dayEMA_combo']

        y_test_sentMA = predictions_dataframe_list_sentMA['actual_price']
        y_test_pred_sentMA = predictions_dataframe_list_sentMA['pred_sentiment_EMA_combo']

        # Print out scores for each model

        print('Brand = {}'.format(df_new.brand[0]))

        print("R2 score for Sentiment only =", round(sm.r2_score(y_test_sent, y_test_pred_sent), 2))

        print("R2 score for 5day Moving Avg only =", round(sm.r2_score(y_test_5dayMA, y_test_pred_5dayMA), 2))

        print("R2 score for Exponential Moving Average only =", round(sm.r2_score(y_test_MA, y_test_pred_MA), 2))

        print("R2 score for 5day Moving Avg & EMA =", round(sm.r2_score(y_test_5dayEMA, y_test_pred_5dayEMA), 2))

        print("R2 score Sentiment & Moving Average =", round(sm.r2_score(y_test_sentMA, y_test_pred_sentMA), 2))

        scrMA_list = np.append(scrMA_list, round(sm.r2_score(y_test_MA, y_test_pred_MA), 2))
        scrMA = scrMA + round(sm.r2_score(y_test_MA, y_test_pred_MA), 2)

        scr5day_list = np.append(scr5day_list, round(sm.r2_score(y_test_5dayMA, y_test_pred_5dayMA), 2))
        scr5day = scr5day + round(sm.r2_score(y_test_5dayMA, y_test_pred_5dayMA), 2)

        scr5dayEMA_list = np.append(scr5dayEMA_list, round(sm.r2_score(y_test_5dayEMA, y_test_pred_5dayEMA), 2))
        scr5dayEMA = scr5dayEMA + round(sm.r2_score(y_test_5dayEMA, y_test_pred_5dayEMA), 2)

        scr_list = np.append(scr_list, round(sm.r2_score(y_test_sentMA, y_test_pred_sentMA), 2))
        scr = scr + round(sm.r2_score(y_test_sentMA, y_test_pred_sentMA), 2)
        print('\n\n\n')
        i = i + 1
        del df_new

        # Set up Dataframes to be added to collection which will make up the ML Model Dataset

        predictions = [predictions_dataframe_list_sent, predictions_dataframe_list_5dayMA,
                       predictions_dataframe_list_5dayEMA,
                       predictions_dataframe_list_MA, predictions_dataframe_list_sentMA]
        df_pred = predictions[0]
        for dfs in predictions[1:]:
            df_pred = df_pred.merge(dfs, left_index=True, right_index=True)

        df_pred = df_pred.drop(columns=['actual_price_x', 'Brand_x', 'actual_price_y', 'Brand_y'])
        full_predictions = full_predictions.append(df_pred)
    else:
        print(
            'Insufficient amount of data to make predictions on {}, there must be at least* 20 days of data. \n\n'.format(
                x))

    # Get overall accuracy scores of each of the models and compare to our sentiment EMA combo model

# Get Averages
acc5dayEMA = round(scr5dayEMA / count, 4)
acc5day = round(scr5day / count, 4)
MAacc = round(scrMA / count, 4)
sentMAacc = round(scr / count, 4)
improve = round(sentMAacc - MAacc, 4)
improve5 = round(sentMAacc - acc5day, 4)
improve5EMA = round(sentMAacc - acc5dayEMA, 4)

print('The amount of companies tested: {}'.format(count))
print('Average accuracy score for 5 day Moving average model = {:.2f}%'.format(acc5day * 100))
print('Average accuracy score for Exponential Moving average Model = {:.2f}%'.format(MAacc * 100))
print('Average accuracy score for Exponential Moving average & 5 Day Model = {:.2f}%'.format(acc5dayEMA * 100))
print('Average accuracy score for Exponential Moving average + Sentiment model = {:.2f}%'.format(sentMAacc * 100))
print('\nOur Sentiment & MA Combination Vs Traditional Models')
print('Model Improvement on 5 day Moving Average = {:.2f}%'.format(improve5 * 100))
print('Model Improvement on Exponential Moving Average = {:.2f}%'.format(improve * 100))

print('Model Improvement on Exponential Moving Average & 5 Day Combo = {:.2f}%'.format(improve5EMA * 100))

# Get Scores
score_list['5day'] = scr5day_list
score_list['5dayEMA'] = scr5dayEMA_list
score_list['MA'] = scrMA_list
score_list['sentMA'] = scr_list
score_list['Brand'] = df_brands

predpath = r'C:\Users\eogha\PycharmProjects\fypBackend\Predictions\prediction_data'
full_predictions.reset_index(drop=True, inplace=True)
full_predictions['id'] = full_predictions.index
full_predictions = full_predictions[
    ['id', 'pred_sentiment', 'pred_5day', 'pred_5dayEMA_combo', 'pred_EMA', 'pred_sentiment_EMA_combo', 'actual_price',
     'Brand']]
from datetime import date

date_time = date.today()


def replacePredData():
    full_predictions.to_csv('{}/PredictedSet-{}.csv'.format(predpath, date_time), encoding='utf-8', index=False)
    conn = connect()
    curr = conn.cursor()
    curr.execute('truncate table fypapp_predictions')
    count = curr.rowcount
    print(count)
    curr.close()
    curr = conn.cursor()

    import csv

    with open('{}/PredictedSet-{}.csv'.format(predpath, date_time), 'r') as f:
        next(f)  # Skip the header row.
        curr.copy_from(f, 'fypapp_predictions', sep=',')
    conn.commit()


replacePredData()

select_query = 'Select * from fypapp_companyinfo'
column_names = ['id', 'ma5score', 'ema5score', 'emascore', 'sentmascore', 'brand', 'ceo', 'revenue', 'industry',
                'location', 'employees']
df = postgresql_to_dataframe(select_query, column_names)

df['ma5score'] = score_list['5day']
df['ema5score'] = score_list['5dayEMA']
df['emascore'] = score_list['MA']
df['sentmascore'] = score_list['sentMA']

from datetime import date

# Replace Accuracy score data

def replaceAccData():
    df.to_csv('{}/CompanyAccuracy-{}.csv'.format(predpath, date_time), encoding='utf-8', index=False, sep=';')
    conn = connect()
    curr = conn.cursor()
    curr.execute('truncate table fypapp_companyinfo')
    count = curr.rowcount
    print(count)
    curr.close()
    curr = conn.cursor()

    import csv

    with open('{}/CompanyAccuracy-{}.csv'.format(predpath, date_time), 'r') as f:
        next(f)  # Skip the header row.
        curr.copy_from(f, 'fypapp_companyinfo', sep=';')
    conn.commit()


replaceAccData()
