import pandas as pd 
import numpy as np 
import datetime 
from dateutil import parser 
import bisect 
import re 

# Date elements
def date_features(df):
    for x in df['date']:
        if '(GMT+' in x :
            df = df.replace({'date': {x: x.split('(')[0]}})

    reg = '[+-][0-9]{2}[0-9]{2}'
    df['time_zone'] = df['date'].apply(lambda x: re.findall(reg, x))
    df['date'] = pd.to_datetime(df['date'], utc = True)
    df['month'] = df['date'].apply(lambda x: str(x.month))
    df['day'] = df['date'].apply(lambda x: str(x.day))
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].apply(lambda x:x.weekday())


    df['time_zone'] = df['time_zone'].apply(lambda x: ['+0000'] if len(x)!= 1 else x)
    df['time_zone'] = df['time_zone'].apply(lambda x: x[0])
    

    df = df.drop(['date'], axis = 1)
    one_hot = pd.get_dummies(df['time_zone'])
    df = df.join(one_hot)

    ## do something with time-zone and check timing

    return df

def mail_features(df) : 
    # clean code a bit
    df['mail_type_0'] = df['mail_type'].apply(lambda x: x.lower().strip().split('/')[0] if type(x) is str else x)
    df['mail_type_1'] = df['mail_type'].apply(lambda x: x.lower().strip().split('/')[1] if type(x) is str else x)
    # mail_type one hot encoding
    one_hot = pd.get_dummies(df['mail_type_0'])
    df = df.join(one_hot)

    one_hot = pd.get_dummies(df['mail_type_1'])
    df = df.join(one_hot)

    df = df.drop(['mail_type_1', 'mail_type_0', 'mail_type'], axis = 1)
    
    return df

# def org_features(df):    
#     df['org'] = df['org'].apply(lambda x: x.lower().strip() if type(x) is str else x)
#     df['org'] = df['org'].apply(lambda x: 'unspecified' if type(x) is not str else x)
#     df['org'] = df['org'].apply(lambda x: 'porn' if type(x) is str and 'porn' in x  else x)
#     df['org'] = df['org'].apply(lambda x: 'centralesupelec' if type(x) is str and ('geeps' in x or 'supelec' in x or 'student-cs' in x) else x)
#     df['org'] = df['org'].apply(lambda x: 'academia' if type(x) is str and ('usief' in x or 'acm' in x or 'iiitd' in x or 'inria' in x or 'researchgatemail' in x or 'researchgate' in x or 'academia' in x or 'ieee' in x or 'slack' in x or 'springboard' in x) else x)
#     df['org'] = df['org'].apply(lambda x: 'social_media' if type(x) is str and ('flickr'in x or 'social' in x or 'facebook' in x or 'twitter' in x or 'quora' in x or 'youtube' in x or 'pinterest' in x or 'medium' in x) else x)
#     df['org'] = df['org'].apply(lambda x: 'online_courses' if type(x) is str and ('nptel' in x or 'udemy' in x or 'mit'in x or 'classroom' in x or 'piazza' in x or 'duolingo' in x or 'learning' in x or 'edx' in x or 'coursera' in x or 'khanacademy' in x or 'usebackpack' in x)else x )
#     df['org'] = df['org'].apply(lambda x: 'google' if type(x) is str and ('google' in x )else x )
#     df['org'] = df['org'].apply(lambda x: 'coding' if type(x) is str and ('mapbox' in x or 'api' in x or 'php' in x or 'tech' in x or 'udacity' in x or 'evernote' in x or 'data' in x or 'stack' in x or 'sharelatex' in x or 'aws' in x or 'codalab' in x or 'trello' in x or 'overleaf' in x  or 'stackexchange' in x or 'hacker' in x or 'code' in x or 'kaggle' in x or 'github' in x or 'nvidia' in x or 'repository' in x ) else x)
#     df['org'] = df['org'].apply(lambda x: 'online_shopping' if type(x) is str and ('ebay' in x or 'amazon' in x) else x)
#     df['org'] = df['org'].apply(lambda x: 'work_media_offer' if type(x) is str and ('cocubes' in x or 'monsterindia' in x or 'job' in x or 'linkedin' in x or 'glassdoor' in x or 'hire' in x or 'career' in x) else x)
#     df['org'] = df['org'].apply(lambda x: 'email' if type(x) is str and ('mail' in x ) else x)
#     df['org'] = df['org'].apply(lambda x: 'travel' if type(x) is str and ('atlassian' in x or 'airfrance' in x or 'airindia' in x or 'airserbia' in x or 'thomascook' in x or 'easyjet' in x or 'ryanair' in x or 'uber' in x) else x)
#     df['org'] = df['org'].apply(lambda x: 'entertainment' if type(x) is str and ('audio' in x or 'audible' in x or 'video' in x or 'imdb' in x or 'cinema' in x or 'entertainment' in x or 'spotify' in x or 'music' in x or 'movies' in x) else x)
#     df['org'] = df['org'].apply(lambda x: 'offers_newsletters' if type(x) is str and ('indiatimes' in x or 'news' in x or 'coupon' in x or 'letter' in x or 'offer' in x)else x)
#     df['org'] = df['org'].apply(lambda x: 'softwares' if type(x) is str and ('app' in x or 'dropbox' in x or 'airtable' in x or 'splitwise' in x)else x)
#     df['org'] = df['org'].apply(lambda x: 'bank' if type(x) is str and ('kotak' in x or 'hsbc' in x or 'paytm' in x)else x)
#     df['org'] = df['org'].apply(lambda x: 'food' if type(x) is str and ('bigbasket' in x or 'food' in x )else x)
#     df['org'] = df['org'].apply(lambda x: 'other' if type(x) is str and (x not in ['food', 'bank', 'softwares', 'offers_newsletters', 'entertainment', 'travel', 'porn', 'centralesupelec', 'academia', 'social_media', 'online_courses', 'google', 'coding', 'online_shopping',  'work_media_offer', 'email'] )else x )

#     one_hot = pd.get_dummies(df['org'])
#     df = df.join(one_hot)

#     df = df.drop(['org'], axis = 1)
    
#     return df

def chars_in_subject_features(df):
    df['chars_in_subject_binned'] = np.searchsorted(list(np.arange(0, 500, 50)), df['chars_in_subject'].values)
    df['chars_in_subject'] = df['chars_in_subject']/df['chars_in_subject'].max()
    return df

# th
def chars_in_body_features(df):
    df['chars_in_body'] = df['chars_in_body']/df['chars_in_body'].max()

    return df

# play around with this
def urls_features(df):
    df['urls_binned'] = np.searchsorted(list(np.arange(0, 500, 25)), df['urls'].values)

    df = df.drop(['urls'], axis = 1)
    
    return df

# def tld_features(df):
#     df['tld'] = df['tld'].apply(lambda x: 'unspecified' if type(x) is not str else x)
    
#     return df

def salutation_designation_features(df):
    df['sals+designation'] = df['salutations'] +  df['designation']

    return df 

def image_features(df): 
    df['images'] = df['images'].apply(lambda x: 'no_images' if x == 0 else 'very_few_images' if  1 <= x <= 15 else 'many_images' if x >= 35 else 'average_images')

    one_hot = pd.get_dummies(df['images'])
    df = df.join(one_hot)

    df = df.drop(['images'], axis = 1)
    return df 

#  Create 3 buckets for time and 2 for day
def hour_flag(df):
    if (16 <= df['hour'] < 24):
        return 'evening_hours'
    elif(0 <= df['hour'] < 8):
        return 'early_morning_hours'
    elif(8 <= df['hour'] < 16):
        return 'work_hours'
    
def hour_features(df):
    df['hour_flag'] = df.apply(hour_flag, axis = 1)

    # do one hot encoding 
    one_hot = pd.get_dummies(df['hour_flag'])
    df = df.join(one_hot)

    df = df.drop(['hour_flag', 'hour'], axis = 1)
    return df 

def day_flag(df):
    if (df['weekday'] <= 5):
        return 0
    else:
        return 1
    
def day_features(df):
    df['weekday_flag'] = df.apply(day_flag, axis = 1)

    df = df.drop(['weekday'], axis=1)
    return df 

    
def tld_features(df):
    df.tld= df.tld.fillna('unspecified')
    df['tld'] = df['tld'].apply(lambda x : x.strip().lower())
    df['tld'] = np.where(df.tld.str.contains('in'), '.in', df.tld)
    df['tld'] = np.where(df.tld.str.contains('org'), '.org', df.tld)
    df['tld'] = np.where(df.tld.str.contains('fr'), '.fr', df.tld)
    df['tld'] = np.where(df.tld.str.contains('com'), '.com', df.tld)
    df['tld'] = np.where(df.tld.str.contains('edu'), '.edu', df.tld)

    tldlist = pd.DataFrame(df.tld.value_counts())
    rarelist = list(tldlist.loc[(tldlist['tld'] < 100)].index)
    df['tld'] = df['tld'].apply(lambda x: 'rare_tld' if x in rarelist else x)

    one_hot = pd.get_dummies(df['tld'])
    df = df.join(one_hot)

    return df

def month_features(df):
    #df['month_2'] = df['date_time'].apply(lambda x: x.strftime("%b" ))
    incremental = ['2','3','4','10','9', '8']
    convex = ['5','6','7']
    concave= ['11','12','1']
    df['month2'] = df['month'].apply(lambda x: 'incremental' if x in incremental else x)
    df['month2'] = df['month'].apply(lambda x: 'convex' if x in convex else x)
    df['month2'] = df['month'].apply(lambda x: 'concave' if x in concave else x)

    one_hot = pd.get_dummies(df['month2'])
    df = df.join(one_hot)
    #df = df.drop('month',axis = 1)

    return df

def org_features(df):
    counts = df['org'].value_counts()
    top13 = pd.DataFrame(counts[:13])
    top13['%'] = top13['org']/ top13['org'].sum()
    top13['%']= pd.to_numeric(top13['%'])

    df.org = df.org.fillna('unspecified_org')

    superlist = list(top13.loc[(top13['%'] >0.3)].index)
    usuallylist = list(top13.loc[(top13['%'] < 0.1) & (top13['%'] > 0.03 )].index)
    oftenlist = list(top13.loc[(top13['%'] < 0.03)].index)
    ls = list(top13.index)
    frequentlist = [x for x in ls if x not in superlist and x not in oftenlist and x not in usuallylist]

    df['org'] = df['org'].apply(lambda x: 'superfrequent_org' if x in superlist else x )
    df['org'] = df['org'].apply(lambda x: 'often_org' if x in oftenlist else x )
    df['org'] = df['org'].apply(lambda x: 'usually_org' if x in usuallylist else x )
    df['org'] = df['org'].apply(lambda x: 'frequent_org' if x in frequentlist else x )

    ls = ['superfrequent_org','often_org','usually_org','frequent_org','unspecified_org']
    df['org'] = df['org'].apply(lambda x: 'rare_org' if x not in ls else x )

    one_hot = pd.get_dummies(df['org'])
    df = df.join(one_hot)
    
    return df 