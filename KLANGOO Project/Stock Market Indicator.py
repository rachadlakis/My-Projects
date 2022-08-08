#!/usr/bin/env python
# coding: utf-8

# # Stock market Indicator Using Klangoo

# ### The stock prices are directly related to people opinion and trust. 
# ### We will use Klangoo to detect negative or positive opinion in certain stocks, in social media or in first class newspapers,  and give a certain indicator to the user if this stock is stable, going up or down according to a certain indicator. 
# 
# ### When surfing  the social media feedbacks (random tweets for example), or  newspaper articles, we will use Klangoo to determine the category of the text :if related to economy, trading, stock market,....( because hashtags are not reliable) and to get main subjects of the texts.
# 
# ### If the category and the subject are related to the stocks we are interested in, we will analyze the sentiment analysis of the texts and create a final indicator to present to the user..
# 
# ### In this notebook, we used a stock market tweets from kaggle, (https://www.kaggle.com/datasets/sohelranaccselab/stock-market-tweets-data-sentiment-analysis ) where each tweet already has the sentiment analysis result (positive, negative, neutral). Each tweet is sent via a request to klangoo then the response is read, if the cateogry of the tweet detected by klangoo is buisness or technology. the tweet will be processed. And if the entities of the tweet contain the entities we are interrested in, the tweet will be processed. A final sentiment analysis will be given to the entities. If the SA is negative, this could be an indicator that a certain stock is going down, and vice versa.
# 

# ### This is a prototype to demonsrate the use of Klangoo in stock indicators

# ### Define a function that will be used in the code

# In[395]:


def have_common(x,y):
   common = False
   for value in x:
      if value in y:
         common= True
   return common


# In[396]:


# Let this the last part, if you have time you can do it.


# In[ ]:





# In[417]:


import pandas as pd
import json
from klangooclient.MagnetAPIClient import MagnetAPIClient

df = pd.read_csv("tweets_labelled.csv") #Read  csv file containig 5000 tweets and the sentiment of each tweet
                                                       # The csv file from kaggle contains alreade the sentiment analysis 
df = df[:30] #50 in testing  #ONly 2000 tweets will be maniupulated bcz we have only 5000 requests from kalngoo


# In[418]:


ENDPOINT ='https://nlp.klangoo.com/Service.svc'
CALK = '016069a5-5b77-4661-b4be-9e71403ab868'
SECRET_KEY = 'SCTXC5kIe/zxeZb3ZD5zg8tcVndcDnUSeIy7jeTF'

client = MagnetAPIClient(ENDPOINT, CALK, SECRET_KEY)


# In[419]:


wanted_entities = ['netflix', 'nflx'] #Stock(s) interrested in, This option is subjective to the user.
wanted_categories = ['general','business', 'politics','technology'] # Only texts having these categoreis will be taken into considertion
                                            # Buisness, and also politics news are very important to the stock prices


# In[420]:


n = len(df)
ctr = 0
grade_sum = 0
final_grade = 0

for i in range(n): #read all the tweets from  the dataFrame
    text = df.iloc[i,2].lower() #get text from dataFrame
    sentiment = df.iloc[i,3]   #get sentiment analysis result from dataFrame
        
    request =  { 'text' : text, 'lang' : 'en', 'format' : 'json' }
    rsp = client.callwebmethod('ProcessDocument', request, 'POST') #Use Klangoo to get all entities and categories of tweet
    info = json.loads(rsp) #transform from Bytes to Json format
    info.fromkeys
    
    entities_nb = len(info['document']['entities'])
    categories_nb = len(info['document']['categories']) 
    
    text_entities = []   # get the key topics from text and append to this list
    text_categories = []   # get the categories from text and append to this list 
    
    for j in range(entities_nb):
        text_entities.append(info['document']['entities'][j]['token'].lower() ) # append the entities to the list
    
    for j in range(categories_nb):
        text_categories.append(info['document']['categories'][j]['name'].lower() ) # append the key categories to the list
    
    print(i)
    print(text)
    print(sentiment)
    print("entities_nb: ", entities_nb)
    print("text_entities: " , text_entities)
    print("text_categories: ", text_categories)
    
    check_category = have_common(wanted_categories, text_categories ) #check existecne of an item of text_categories in wanted_category 
    check_entities = have_common(wanted_entities, text_entities )
    print("check_category: ", check_category)
    print("check_entities: " , check_entities)
    print("========")
    
    if check_category  and check_entities: # if the category and topics of the text are in what we are interrested in 
        print("Final step")
        grade = 0
        if sentiment == 'positive':
            grade = 5
        elif sentiment == 'negative':
            grade = -5
        #else is: stays zero 
        ctr += 1
        grade_sum += grade   
  

if ctr != 0:
    final_grade = grade_sum / ctr

    if final_grade <0:
        print("The Netflix stock have negative Semantic Analysis in the given tweets, Think of selling. After         processing ", ctr , " tweets, the term Netfilx or NFLX got a scrore of : " , final_grade, " on 5 .")
    elif final_grade >0:
        print("The Netflix stock have a positive Semantic Analysis in the given tweets, Think of buying or hloding. After         processing ", ctr , " tweets, the term Netfilx or NFLX got a scrore of : " , final_grade, " on 5 .")
    else:
        print("The Netflix stock have a neutral Semantic Analysis. ")  


# In[ ]:


#When the entity is amazon the klangoo is giving empty string as response


# In[ ]:





# In[ ]:





# # Testing

# In[413]:


from klangooclient.MagnetAPIClient import MagnetAPIClient
import json
request = { 'text' : 'RT @gzbenso: Netflix leads its rivals in original TV shows by a wide margin in both quantity and quality  according to new data analysis htâ€¦ ',          'lang' : 'en', 'format' : 'json' }

rsp = client.callwebmethod('ProcessDocument', request, 'POST')
info =json.loads(rsp) #transform from Bytes to Json format
info.fromkeys

#get all the key topics from the text
entities_nb = len(info['document']['entities'])
categories_nb = len(info['document']['categories'])
 

entities = [] # get the key topics from text and append to this list
categories = [] # get the categories from text and append to this list
 

for i in range(entities_nb):
    entities.append(info['document']['entities'][i]['token'].lower() )
    
for i in range(categories_nb):
    categories.append(info['document']['categories'][i]['name'].lower())
    

    
print( entities)  
print( categories)


# In[368]:


print("text_entities: " , text_entities)  
print("text_categories: " , text_categories)
print("key_topics: " , key_topics)


# In[365]:


check_category = any(item in wanted_categories for item in categories)#check existecne of an item of text_categories in wanted_category 
check_topics = any( item in entities for item in user_important_stocks )
print("catogories: " , categories)
print("wanted_categories: " , wanted_categories)

print("entities: ", entities)
print("user_important_stocks: ", user_important_stocks)

print(check_category, check_topics)


# In[366]:


check_category = any(item in wanted_categories for item in text_categories)#check existecne of an item of text_categories in wanted_category 
check_entities = any( item in user_important_stocks for item in text_entities )
if check_category  and check_entities: # if the category and topics of the text is in what we are interrested in 
    print("Final step")
    grade = 0
    if sentiment =='positive':
        grade = 5
    elif sentiment =='negative':
        grade = -5
    #else is stays zero 
    ctr += 1
    grade_sum += grade
print("ctr: " , ctr)
print("grade_sum: " ,grade_sum)

print(text_categories)
print(text_entities)


# In[280]:


text_categories


# In[349]:


df.iloc[2,]


# In[ ]:




