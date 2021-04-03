import re
from tweepy import OAuthHandler
import twitter_credentials
import tweepy
import json
# OAuth
auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

search_terms = ['Activision',
                'Adidas',
                'AIG',
                'Amazon',
                'Apple',
                'AT&T',
                'CVS',
                'Dell',
                'Disney',
                'EA',
                'EasyJet',
                'Exxon',
                'Ford',
                'Gamestop',
                'Tesla',
                'AMC',
                'BlackBerry',
                'Hertz',
                'Intel',
                'Lowes',
                'Microsoft',
                'Netflix',
                'Nike',
                'Nintendo',
                'Pepsi',
                'Pfizer',
                'Samsung',
                'Sony',
                'Starbucks',
                'Ubisoft',
                'Walmart'] # Array containing all current companies I'm analysing

def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', tweet).split())


def stream_tweets(search_term):
    data = []  # empty list to which tweet_details obj will be added
    counter = 0  # counter to keep track of each iteration
    for tweet in tweepy.Cursor(api.search, q='\"{}\" -filter:retweets'.format(search_term), count=100, lang='en',
                               tweet_mode='extended').items():

        tweet_details = {}
        tweet_details['tag'] = search_term
        tweet_details['tweet'] = tweet.full_text
        tweet_details['created'] = tweet.created_at.strftime("%d-%b-%Y")  # Want timestamps to match to stock price
        clean_t = clean_tweet(tweet_details['tweet'])  # Gets rid of hyperlinks
        tweet_details['tweet'] = clean_t
        print(search_term)
        data.append(tweet_details)
        counter += 1
        if counter == 100:  # Api.search only runs 100 times so add a loop
            break
        else:
            pass
    with open('Twot/data/{}.json'.format(search_term), 'a') as f:
        json.dump(data, f)
    print('done!')




if __name__ == "__main__":

    print('Starting to stream...')
    counter = 0
    while True:  # Infinite loop
        for search_term in search_terms:
            stream_tweets(search_term)
            print('finished!')
            counter += 1

