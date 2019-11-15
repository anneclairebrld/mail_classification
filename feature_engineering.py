import pandas as pd 
import numpy as np 
import datetime 
from dateutil import parser 
import bisect 

# Date elements
def date_features(df):
    for x in df['date']: # get rid of (GMT+) elements -> already encoded
            if '(GMT+' in x :
                df = df.replace({'date': {x: x.split('(')[0]}})


    # df['date'] = pd.to_datetime(df['date'], utc = True)
    # df['month'] = df['date'].dt.month
    # df['year'] = df['date'].dt.year
    # df['day'] = df['date'].dt.date
    # df['hour'] = df['date'].dt.hour

    df['date_time'] = [parser.parse(x) for x in df['date']]
    df['time'] = [str(x.time()) for x in df['date_time']]
    df['time_zone'] = [str(x)[-6:] for x in df['date_time']]
    df['year'] = [x.date().year for x in df['date_time']] 
    df['month'] = [x.date().month for x in df['date_time']] 
    df['day'] = [x.date().day for x in df['date_time']] 
    df['weekday'] = [x.date().weekday() for x in df['date_time']]
    
    one_hot = pd.get_dummies(df['weekday'])
    df = df.join(one_hot)
    return df

def mail_features(df) : 
    # clean code a bit
    df['mail_type'] = df['mail_type'].apply(lambda x: x.lower().strip() if type(x) is str else x)
    
    # mail_type one hot encoding
    one_hot = pd.get_dummies(df['mail_type'])
    df = df.join(one_hot)
    
    return df

def org_features(df):    
    df['org'] = df['org'].apply(lambda x: x.lower().strip() if type(x) is str else x)
    df['org'] = df['org'].apply(lambda x: 'unspecified' if type(x) is not str else x)
    df['org'] = df['org'].apply(lambda x: 'porn' if type(x) is str and 'porn' in x  else x)
    df['org'] = df['org'].apply(lambda x: 'centralesupelec' if type(x) is str and ('geeps' in x or 'supelec' in x or 'student-cs' in x) else x)
    df['org'] = df['org'].apply(lambda x: 'academia' if type(x) is str and ('usief' in x or 'acm' in x or 'iiitd' in x or 'inria' in x or 'researchgatemail' in x or 'researchgate' in x or 'academia' in x or 'ieee' in x or 'slack' in x or 'springboard' in x) else x)
    df['org'] = df['org'].apply(lambda x: 'social_media' if type(x) is str and ('flickr'in x or 'social' in x or 'facebook' in x or 'twitter' in x or 'quora' in x or 'youtube' in x or 'pinterest' in x or 'medium' in x) else x)
    df['org'] = df['org'].apply(lambda x: 'online_courses' if type(x) is str and ('nptel' in x or 'udemy' in x or 'mit'in x or 'classroom' in x or 'piazza' in x or 'duolingo' in x or 'learning' in x or 'edx' in x or 'coursera' in x or 'khanacademy' in x or 'usebackpack' in x)else x )
    df['org'] = df['org'].apply(lambda x: 'google' if type(x) is str and ('google' in x )else x )
    df['org'] = df['org'].apply(lambda x: 'coding' if type(x) is str and ('mapbox' in x or 'api' in x or 'php' in x or 'tech' in x or 'udacity' in x or 'evernote' in x or 'data' in x or 'stack' in x or 'sharelatex' in x or 'aws' in x or 'codalab' in x or 'trello' in x or 'overleaf' in x  or 'stackexchange' in x or 'hacker' in x or 'code' in x or 'kaggle' in x or 'github' in x or 'nvidia' in x or 'repository' in x ) else x)
    df['org'] = df['org'].apply(lambda x: 'online_shopping' if type(x) is str and ('ebay' in x or 'amazon' in x) else x)
    df['org'] = df['org'].apply(lambda x: 'work_media_offer' if type(x) is str and ('cocubes' in x or 'monsterindia' in x or 'job' in x or 'linkedin' in x or 'glassdoor' in x or 'hire' in x or 'career' in x) else x)
    df['org'] = df['org'].apply(lambda x: 'email' if type(x) is str and ('mail' in x ) else x)
    df['org'] = df['org'].apply(lambda x: 'travel' if type(x) is str and ('atlassian' in x or 'airfrance' in x or 'airindia' in x or 'airserbia' in x or 'thomascook' in x or 'easyjet' in x or 'ryanair' in x or 'uber' in x) else x)
    df['org'] = df['org'].apply(lambda x: 'entertainment' if type(x) is str and ('audio' in x or 'audible' in x or 'video' in x or 'imdb' in x or 'cinema' in x or 'entertainment' in x or 'spotify' in x or 'music' in x or 'movies' in x) else x)
    df['org'] = df['org'].apply(lambda x: 'offers_newsletters' if type(x) is str and ('indiatimes' in x or 'news' in x or 'coupon' in x or 'letter' in x or 'offer' in x)else x)
    df['org'] = df['org'].apply(lambda x: 'softwares' if type(x) is str and ('app' in x or 'dropbox' in x or 'airtable' in x or 'splitwise' in x)else x)
    df['org'] = df['org'].apply(lambda x: 'bank' if type(x) is str and ('kotak' in x or 'hsbc' in x or 'paytm' in x)else x)
    df['org'] = df['org'].apply(lambda x: 'food' if type(x) is str and ('bigbasket' in x or 'food' in x )else x)
    df['org'] = df['org'].apply(lambda x: 'other' if type(x) is str and (x not in ['food', 'bank', 'softwares', 'offers_newsletters', 'entertainment', 'travel', 'porn', 'centralesupelec', 'academia', 'social_media', 'online_courses', 'google', 'coding', 'online_shopping',  'work_media_offer', 'email'] )else x )

    one_hot = pd.get_dummies(df['org'])
    df = df.join(one_hot)
    
    return df

# play around with this
def chars_in_subject_features(df):
    df['chars_in_subject_binned'] = np.searchsorted(list(np.arange(0, 500, 25)), df['chars_in_subject'].values)
    
    return df

# play around with this
def urls_features(df):
    df['urls_binned'] = np.searchsorted(list(np.arange(0, 500, 25)), df['urls'].values)
    
    return df

def tld_features(df):
    df['tld'] = df['tld'].apply(lambda x: 'unspecified' if type(x) is not str else x)
    
    return df