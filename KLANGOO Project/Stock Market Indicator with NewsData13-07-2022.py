#!/usr/bin/env python
# coding: utf-8

# # Stock market Indicator Using Klangoo

# ### The stock prices are directly related to people opinion and trust. Sentiment Analysis can be used to determine the overall opinion for a certain stock. 
# 
# ### A fast answer can be to apply sentiment analysis on every text contaning the word(s) we are interrested in.
# 
# ### But, It is not accurate to study every text containing the keyword we are looking for, si how de we know if the domain of the text is  the same domain we we are interrested in (Buisness, Technology, ...) or the text is about a completely different domain and citing this word as an example while speaking about another thing??  
# 
# ### How do we know if a certain text is directly related to the stock we are interrested in (i.e.: the stock is the main subject or one of the main subjects of the text) and not cited as a secondary subject in the text?
# 
# ## KLANGOO can be the answer
# 
# ### Klangoo can be used to determine, for each text (twitter tweet or a newspaper article) if the domain and the main subject of the text (entities) are what we are interrested in. 
# 
# 
# ### In the first part, we used a dataset already containng the sentiment analysis results of tweets. We used a stock market tweets from kaggle, (https://www.kaggle.com/datasets/sohelranaccselab/stock-market-tweets-data-sentiment-analysis ).
# ### Each row of the csv file contains a tweet details, the tweet initial text and a sentiment analysis result (positive, negative, neutral) .
# ### After sending the tweet as a requeest to Klangoo, we process the results: if the main category and the main entities of the tweet are appropriate: the tweet will be processed. Otherwise it will be neglected.
# ### A final sentiment analysis will be given to the wanted entity. If it is negative, this could be an indicator that the  stock is going down (maybe, nothing is 100% sure in stock market), and vice versa.
# 
# ### The result will be a simple advice of selling, or keeping / buying a certain stock according to the general public opinion. Also a final grade over 5 will be given with the number of tweets analysed to give a certain inficator to the user.
# 
# ### In the second part (real use case) an API was used to collect articles from different sources. The api results of giving atricles of a main subject and of a certain language. The results are sent one by one to Klangoo to speicify, as in first part, the main categories and entities.
# 

# ### Define a function that will be used in the code

# In[330]:


def have_common(x,y):
   common = False
   for value in x:
      if value in y:
         common= True
   return common


# ### Importing Libraries and reading the csv file a Pandas dataFrame

# In[331]:


from klangooclient.MagnetAPIClient import MagnetAPIClient
import pandas as pd
import json
import string

df = pd.read_csv("tweets_labelled.csv") #Read  csv file containig 5000 tweets and the sentiment of each tweet
                                                       # The csv file from kaggle contains alreade the sentiment analysis 
df = df[:5] #only 30 rows were processed 
print()


# ### Prepare connection strings and secret key

# In[152]:


ENDPOINT ='https://nlp.klangoo.com/Service.svc'
CALK = '155b99fc-758c-40c8-a8ef-5d4eb9fba616'
SECRET_KEY = 'wTMeUVHL38vGrdDBbSdSFBQOWFchPTaUnIFZBJS4'

client = MagnetAPIClient(ENDPOINT, CALK, SECRET_KEY)


# ### Specify wanted entities and categories to search for

# In[159]:


wanted_stock = "Amazon stock AMZN"
wanted_entities = ['amazon', 'amzn'] #Stock(s) interrested in, This option is subjective to the user.
wanted_categories = [ 'science','general','business', 'politics','technology'] #Only these categoreis will be taken into considertion


# ### Code

# In[160]:


n = len(df)
ctr = 0
grade_sum = 0
final_grade = 0

for i in range(n): #read all the tweets from  the dataFrame
    text = df.iloc[i,2] #get text from dataFrame
    text = text
    sentiment = df.iloc[i,3]   #get sentiment analysis result from dataFrame
        
    request =  { 'text' : text, 'lang' : 'en', 'format' : 'json' }
    
    rsp = client.callwebmethod('ProcessDocument', request, 'POST') #Use Klangoo to get all entities and categories of tweet
    info = json.loads(rsp) #transform to Dictionary
    info.fromkeys
    print("request nb:", i , "done")
    print("Text for request nb:", i, "is: ", text)
    print("Sentiment for request nb:", i, "is: ", sentiment)

    entities_nb = len(info['document']['entities'])
    categories_nb = len(info['document']['categories']) 
    
    text_entities = []   # get the key topics from text and append to this list
    text_categories = []   # get the categories from text and append to this list 
    
    for j in range(entities_nb):
        text_entities.append(info['document']['entities'][j]['token'].lower())#adding categoreis from klangoo response to  text_categories
    
    for j in range(categories_nb):
        text_categories.append(info['document']['categories'][j]['name'].lower())#adding categoreis from klangoo response to  text_categories
    
    print("Entities for request nb:", i, "are: ", text_entities)
    print("Categories for request nb:", i, "are: ", text_categories, "\n")

    check_category = have_common(wanted_categories, text_categories ) #check existecne of an item of text_categories in wanted_category 
    check_entities = have_common(wanted_entities, text_entities )
    
    if check_category  and check_entities: # if the category and topics of the text are in what we are interrested in 
        if sentiment == 'positive':
            grade_sum += 1   
        elif sentiment == 'negative':
            grade_sum -= 1
        #else is: stays zero 
        ctr += 1
           
  

if ctr != 0:
    final_grade = grade_sum / ctr

    if final_grade <0:
        print("The " , wanted_stock  ,"have negative Semantic Analysis in the given tweets, it could be an indicator         that this stock does not have community trust in the dtermined time.\n  After         processing ", ctr , " tweets, the term Netfilx or NFLX got a scrore of : " , final_grade, " on 1 .")
    elif final_grade >0:
        print("The ",wanted_stock, "  have a positive Semantic Analysis in the given tweets, it could be an indicator         that this stock have community trust in the dtermined time.\n  After         processing ", ctr , " tweets, the term Netfilx or NFLX got a scrore of : " , final_grade, " on 1 .")
    else:
        print("After analysing the specifeid texts (tweets), The ", wanted_stock, " have a neutral Semantic Analysis. ")  


# ## Second Part: Real Use case (Bitcoin)
# 
# ### Now, after we tried our idea on a dataset, let's try our idea on a real use case. We will get news articles from different newspapers and apply the sentiment analysis and use klangoo to make a final indicator only for the stocks we are interrested in.
# 
# ### The newsapi is used here to provide articles from different sources (Better than searching in hundreds of websites) containig a specific word and in a certain date range (from -to), in ceratin websites we specify or over 80,000 websites.
# 
# ### But it is not logical to take all articles containng this word into consideration. So we will use Klangoo to know the main entities and the domain of the article, then we can apply sentiment analysis.
# 

# ###

# ## Bitcoin stock Indicator between 01/07/2022 and 13/07/2022

# In[332]:


import requests

#get all the articles from .. to .. containin the words realted to bicoin
url = ('https://newsapi.org/v2/everything?' 
       'q=Bitcoin USD& Or q=BTC-USD& or q=bitcoin& or q=Bitcoin& or q=BTC& or q=btc&'  #serach for articles containng ... word
       'from=2022-07-01&'     #search article in the last week 
       'to=2022-07-13&'
       'language=en&'
       'sortBy=popularity&'
       'apiKey=42f6576050c04962b5556e40825857df')

response = requests.get(url)
print("Request successful")


# In[333]:


pip install newsdataapi


# In[334]:


from newsdataapi import NewsDataApiClient

api = NewsDataApiClient(apikey='pub_9173b9c5020815c947c4ebd4e19094a1b11d')
#https://newsdata.io/

response = api.news_api( q= "bitcoin OR Bitcoin OR bitcoins OR Bitcoins", language='en')
print("Done")


# In[335]:


print(len(response['results']))


# In[336]:


print("First text:" , response['results'][0]['content'])


# In[337]:


#determine the wanted entity and categories
wanted_stock_2 = "Bitcoin stock BTC"
wanted_entities_2 = ['bitcoin', 'btc'] #Stock(s) interrested in, This option is subjective to the user.
wanted_categories_2 = ['general','business', 'politics','technology']#Only these categoreis will be taken into considertion


# In[338]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

nb = len(response['results'])
temp_nb = 7 #For testing only
ctr_2 = 0
grade_sum_2 = 0
final_grade_2 = 0

for i in range(temp_nb):
    #Since the API give only 200 chars in contnet, this is not acceptable
    # a good idea would be to use the title and the description as a subsitute since they contion most of the meaning.
    
    text_2 = response['results'][i]['content']
    if text_2 is not None:
        request_2 =  { 'text' : text_2, 'lang' : 'en', 'format' : 'json' }
        rsp_2 = client.callwebmethod('ProcessDocument', request_2, 'POST') #Use Klangoo to get all entities and categories of tweet
        info_2 = json.loads(rsp_2) #transform from Bytes to Json format
        info_2.fromkeys

        print("Request nb:",i,"is successful.")
        print("Text nb:", i, "is: ", text_2)

        entities_nb_2 = len(info_2['document']['entities'])
        categories_nb_2 = len(info_2['document']['categories']) 

        text_entities_2 = []   # get the key topics from text and append to this list
        text_categories_2 = []   # get the categories from text and append to this list 

        for j in range(entities_nb_2):#adding entites from klangoo response to  text_entities_2 list
            text_entities_2.append(info_2['document']['entities'][j]['token'].lower() ) 

        for j in range(categories_nb_2):#adding categoreis from klangoo response to  text_categories_2
            text_categories_2.append(info_2['document']['categories'][j]['name'].lower() ) # append the key categories to the list

        print("Entities for Text nb:", i, "are: ", text_entities_2)
        print("Categories for Text nb:", i, "are: ", text_categories_2, "\n")
        
        check_category_2 = have_common(wanted_categories_2, text_categories_2 ) #check existecne of an item of text_categories in wanted_category 
        check_entities_2 = have_common(wanted_entities_2, text_entities_2 )

        if check_category_2  and check_entities_2: #If the article category and entity are the wanted ones, we apply S.A.
            ctr_2 += 1
            score_2 = analyser.polarity_scores(text_2)
            print("Sentiment analysis for text: ",  i , "is: ", score_2)
            grade_sum_2 += score_2['compound'] #compound element of the polarity score determine if positive >0.5 or negative <-0.5
        print("check_category condition: ", check_category_2 )
        print("check_entities condition: ", check_entities_2)
        print()
        print("-----------------")
        print()
        

print("counter", ctr_2)        
if ctr_2 > 0:
    final_grade_2 = grade_sum_2 / ctr_2

    if final_grade_2 < -0.1: #check the condition neg or pos and total score in NLTK
        print("The " , wanted_stock_2  ,"have negative Semantic Analysis in the given tweets,it could be an indicator         that this stock does not have community trust in the dtermined time.        After processing ", ctr_2 , " articles, the term bitcoin or BTC got a polarity scrore of : " , final_grade_2, " on 1 .")
    elif final_grade > 0.1:
        print("The ",wanted_stock_2, "  have a positive Semantic Analysis in the given tweets,it could be an indicator         that this stock have community trust in the determined time.         After processing ", ctr_2 , " articles,the term bitcoin or BTC got a polarity scrore of :" , final_grade_2, " on 1 .")
    else:
        print("After analysing the relative articles, the ", wanted_stock_2, " have a neutral Semantic Analysis.")          
    

    


# In[ ]:




