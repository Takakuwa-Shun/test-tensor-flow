# -*- coding:utf-8 -*-

import tweepy
import pandas as pd

# 各種キーをセット
CONSUMER_KEY = 'KqflgH5fCIrrInHCYbFMpdJSs'
CONSUMER_SECRET = 'lYK7jpE1zOmXrE2kv9WYVy9soxtIVw4eTsCc1alFLB9F7YvnOz'
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
ACCESS_TOKEN = '1409470201-6mP512G0j4BTWs26xO2GqFFwJqEoXbYXRDjs79N'
ACCESS_SECRET = '9sCaatuhNno4HqwzOYjQUfDCi6oNvXy31nMtSZq41Kewh'
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

#APIインスタンスを作成
api = tweepy.API(auth)

# print(api.home_timeline()[0].text)


Tweet = api.user_timeline("@Suzu_Mg")

Num_Tweet = 20
index = [i for i in range(Num_Tweet)]
Hirose_Tweet =[]
for i in range(Num_Tweet):
    Hirose_Tweet += [Tweet[i].text]
df_tweet = pd.DataFrame(Hirose_Tweet, index=index, columns= "広瀬すず")
print(df_tweet)