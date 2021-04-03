import glob
import ntpath
import os
import datetime
from datetime import datetime
from datetime import date
import calendar
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas_datareader.data as web
import sys
import pandas as pd
import psycopg2

# Paths for data files
aggpath = r'AggData\aggdata'
aggdatapath = r'AggData\data'
twotpath = r'Twot\data'

date_time = date.today()


# Postgres Functions
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


def postgresql_to_dataframe(conn, select_query, column_names):
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


def replaceData():
    conn = connect()
    curr = conn.cursor()
    curr.execute('truncate table fypapp_aggdata')
    count = curr.rowcount
    print(count)
    curr.close()
    curr = conn.cursor()

    import csv

    with open('{}/AggregatedSet-{}.csv'.format(aggpath, date_time), 'r') as f:
        next(f)  # Skip the header row.
        curr.copy_from(f, 'fypapp_aggdata', sep=',')
    conn.commit()


connect()

# Get the raw data
for filepath in glob.iglob(r'{}\*.json'.format(twotpath)):
    # print(filepath)
    f = open('{}'.format(filepath), 'r')
    filedata = f.read()
    f.close()

    filename = ntpath.basename(filepath)

    newdata = filedata.replace('][', ',')
    f = open('{}/{}'.format(aggdatapath, filename), 'w+')
    f.write(newdata)
    f.close()
    del filedata

# Files now ready for converting into DataFrames and put into a collection

df_collection = {}
i = 0

for filepath in glob.iglob(r'{}/*.json'.format(aggdatapath)):
    print(filepath)
    with open(filepath) as f:
        df_collection[i] = pd.read_json(filepath)
    i = i + 1
    f.close()

# Analyse new data
print("Starting Analyser \n\n")
for dfs in df_collection:
    print("\nAnalysing... ")
    # VADER SENTIMENT ANALYSER
    analyzer = SentimentIntensityAnalyzer()

    df_collection[dfs]['neg'] = df_collection[dfs]['tweet'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
    df_collection[dfs]['neu'] = df_collection[dfs]['tweet'].apply(lambda x: analyzer.polarity_scores(x)['neu'])
    df_collection[dfs]['pos'] = df_collection[dfs]['tweet'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
    df_collection[dfs]['compound'] = df_collection[dfs]['tweet'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'])

    print(df_collection[dfs].head())

print("\n\nFinished Analysing")


# Normalise the dates
def normaldate(dte):
    # print('Altering date format')
    dte = dte.split('-')
    day = dte[0]
    month_ab = dte[1]
    year = dte[2]
    month = list(calendar.month_abbr).index(month_ab)
    datestr = day + str(month).zfill(2) + year
    # print(datestr)
    fulldate = datetime.strptime(datestr, '%d%m%Y').date()
    return fulldate


for dfs in df_collection:
    name = df_collection[dfs].tag[0]
    print('Altering dates for ' + name)
    df_collection[dfs]['date'] = df_collection[dfs].created.apply(normaldate)
    print('Finished normalising dates for {}\n'.format(name))

# Aggregates some of the new data

# Tickers. Future work job: Get tickers from DB. Hardcode bad
ticker = ['ATVI',
          'ADS.DE',
          'AIG',
          'AMZN',
          'AMC',
          'AAPL',
          'T',
          'BB',
          'CVS',
          'DELL',
          'DIS',
          'EA',
          'EZJ',
          'XOM',
          'F',
          'GME',
          'HTZGQ',
          'INTC',
          'LOW',
          'MSFT',
          'NFLX',
          'NKE',
          'NTDOY',
          'PEP',
          'PFE',
          '005930.KS',
          'SNE',
          'SBUX',
          'TSLA',
          'UBSFF',
          'WMT']  # Contains all of the tickers for the companies were analysing

# Aggregate the data and get stock price for the day
aggset = {}
stock = pd.DataFrame(columns=['Date', 'closeprice'])
for dfs in df_collection:
    brand = df_collection[dfs].tag[0]
    comp = df_collection[dfs].groupby('date')[['compound']].mean()
    pos = df_collection[dfs].groupby('date')[['pos']].mean()
    neg = df_collection[dfs].groupby('date')[['neg']].mean()
    neu = df_collection[dfs].groupby('date')[['neu']].mean()

    agg_data = pd.DataFrame(pos)
    agg_data['neg'] = neg
    agg_data['neu'] = neu
    agg_data['compund'] = comp
    agg_data['brand'] = brand
    agg_data['Date'] = agg_data.index
    agg_data['ticker'] = ticker[dfs]
    agg_data['closeprice'] = 0
    days = agg_data.index.tolist()

    aggset[dfs] = agg_data

print(aggset)
print('Finished that!')

fullset = aggset
# Get the Close Price for each company. Important: Schedule for after market closed
for dfs in fullset:

    for i in range(0, len(fullset[dfs])):
        start = fullset[dfs]['Date'].iloc[i]

        print(type(start))
        try:
            df_reader = web.DataReader(fullset[dfs]['ticker'].iloc[0], 'yahoo', start, start)
            closeprice = df_reader['Adj Close']

            if fullset[dfs].brand[i] == 'Samsung':
                print('Converting Korean to USD for Samsung')
                closeprice[0] = (closeprice[0] * .0009)
            fullset[dfs]['closeprice'].iloc[i] = closeprice[0]
        except KeyError:
            pass

# Get  current data so we can deal with nan days of data

df_current = pd.concat(fullset)
df_current.columns = ['pos', 'neg', 'neu', 'comp', 'brand', 'day', 'ticker', 'closeprice']

column_names = ['pos', 'neg', 'neu', 'comp', 'brand', 'day', 'ticker', 'closeprice']

# Get Data from DB
df_all = postgresql_to_dataframe(0, 'select pos,neg,neu,comp,brand,day,ticker,closeprice from fypapp_aggdata',
                                 column_names)
df_current.reset_index(drop=True, inplace=True)
df_all = df_all.append(df_current)
df_all.reset_index(drop=True, inplace=True)

# Needs to be sorted back into dictionary
UniqueNames = df_all.brand.unique()

DataFrameDict = {elem: pd.DataFrame for elem in UniqueNames}

for key in DataFrameDict.keys():
    DataFrameDict[key] = df_all[:][df_all.brand == key]

df_dict = DataFrameDict

dic = dict()
for key in df_dict:
    if not df_dict[key].empty:
        dic[key] = df_dict[key]

fullset = dic

# At this point there is a dictionary each element is the agg data for that day and the stock close price
# Next i'll need to deal with gaps in the data from days the market is closed

for dfs in fullset:
    fullset[dfs].reset_index(drop=True, inplace=True)
    fullset[dfs].closeprice = fullset[dfs].closeprice.fillna(0)

    # CHANGE 1ST VALUE - WORKS
    # --------------------------------------------------------------------------------------------------
    count = 0
    j = 0
    x = 0
    y = 0
    z = 0
    if fullset[dfs].closeprice[0] == 0:
        print('1st Val is nan for the {} collection')
        while x == 0 or y == 0:
            j = j + 1
            if fullset[dfs].closeprice[j] != 0:
                print('possible value')
                if x == 0:
                    x = fullset[dfs].closeprice[j]
                elif y == 0:
                    y = fullset[dfs].closeprice[j]
        z = (x + y) / 2
        fullset[dfs].closeprice[0] = z
        print(fullset[dfs].closeprice)

    # CHANGE LAST VALUE - WORKS
    # -------------------------------------------------------------------------------------------------
    size = len(fullset[dfs].closeprice) - 1
    # print(size)
    count = 0
    j = size
    x = 0
    y = 0
    z = 0
    if fullset[dfs].closeprice[size] == 0:
        # print('last Val is nan')
        while x == 0 or y == 0:
            j = j - 1
            if fullset[dfs].closeprice[j] != 0:
                # print('possible value')
                if x == 0:
                    x = fullset[dfs].closeprice[j]
                elif y == 0:
                    y = fullset[dfs].closeprice[j]
        z = (x + y) / 2
        fullset[dfs].closeprice[size] = z
        # print(fullset[dfs].closeprice)

    # AVERAGE OUT ALL REMAINING GAPS - WORKS
    # -----------------------------------------------------------------------------------------
    k = 0
    x = 0
    y = 0
    z = 0
    i = 1
    size = len(fullset[dfs].closeprice) - 1
    for i in range(size):
        print('current value is: {} at  position   {}'.format(fullset[dfs].closeprice[i], i))
        # print('\n')

        if fullset[dfs].closeprice[i] == 0:
            print('this needs fixin')

            j = i
            while fullset[dfs].closeprice[j] == 0:
                j = j - 1
            print('Found lower half: {} at  position   {}'.format(fullset[dfs].closeprice[j], j))
            x = fullset[dfs].closeprice[j]

            k = i
            while fullset[dfs].closeprice[k] == 0:
                k = k + 1
            print('Found upper half: {} at  position   {}'.format(fullset[dfs].closeprice[k], k))
            y = fullset[dfs].closeprice[k]

            z = (x + y) / 2
            print('The Average for this value i : {}'.format(z))
            fullset[dfs].closeprice[i] = z
            print('\n')

        print(fullset[dfs])

for dfs in fullset:
    size = len(fullset[dfs].brand)
    print(size)
    name = fullset[dfs].brand[0]
    print(fullset[dfs].head())

    for i in range(0, size):
        if fullset[dfs].brand[i] != fullset[dfs].brand[0]:
            fullset[dfs].brand[i] = fullset[dfs].brand[0]

        if fullset[dfs].ticker[i] != fullset[dfs].ticker[0]:
            fullset[dfs].ticker[i] = fullset[dfs].ticker[0]

        # .0009 is exchange rate korean to usd - not an ideal fix
        # Future Work: Change Company List to NASDAQ 100
        if fullset[dfs].brand[i] == 'Samsung' and fullset[dfs].closeprice[i] > 1000:
            print('Converting Korean to USD for Samsung')
            fullset[dfs].closeprice[i] = (fullset[dfs].closeprice[i] * .0009)

df_current = pd.concat(fullset)
date_time = fullset['Activision'].day[len(fullset['Activision']) - 1].strftime("%d-%m-%Y")
print("date and time:", date_time)

df_current.reset_index(drop=True, inplace=True)
df_alldata = df_current

df_alldata.sort_values(by='brand')
UniqueNames = df_alldata.brand.unique()
DataFrameDict = {elem: pd.DataFrame for elem in UniqueNames}

for key in DataFrameDict.keys():
    DataFrameDict[key] = df_alldata[:][df_alldata.brand == key]

    df_dict = DataFrameDict
    dic = dict()
    for key in df_dict:
        if not df_dict[key].empty:
            dic[key] = df_dict[key]

df_dict = dic

# Some columns need to be altered here as new data is available
# Get the moving averages

for key in df_dict:
    df = df_dict[key]
    print(df.brand)
    print('Calculating additional data...')
    df['5dayMA'] = df['closeprice'].rolling(window=5).mean()
    df['EMA'] = df['closeprice'].ewm(span=8).mean()

    # Getting future days - not used
    # Future work: Potentially remove future days

    for key in df_dict:
        i = 0
        df = df_dict[key]
        size = len(df)
        future = 6
        df['closeprice'] = df['closeprice'].round(2)
        for f in range(1, future):
            df['fut{}'.format(f)] = 0
            for i in range(size - f):
                df['fut{}'.format(f)].iloc[i] = df['closeprice'].iloc[i + f]
        df.fillna(0)

df_alldata = pd.concat(df_dict)
df_alldata.reset_index(drop=True, inplace=True)
df_alldata['id'] = df_alldata.index
df_alldata = df_alldata[
    ['id', 'pos', 'neg', 'neu', 'comp', 'brand', 'day', 'ticker', 'closeprice', '5dayMA', 'EMA', 'fut1', 'fut2', 'fut3',
     'fut4', 'fut5']]
df_alldata = df_alldata.fillna(0)

# Save to local filesystem before adding to database for safety

df_alldata.to_csv('{}/AggregatedSet-{}.csv'.format(aggpath, date_time), encoding='utf-8', index=False)
connect()

# Update DB

replaceData()

# NEED TO CLEAN UP DATA FILES SO THIS PROCESS CAN START AGAIN IN 24 HRS
for filepath in glob.iglob(r'{}\*.json'.format(twotpath)):
    os.remove(filepath)
for filepath in glob.iglob(r'{}\*.json'.format(aggdatapath)):
    os.remove(filepath)

