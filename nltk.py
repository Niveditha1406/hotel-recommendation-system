import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval
import pandas as pd
data = pd.read_csv("/content/Hotel_Reviews1.csv")
print(data.Hotel_Address.head())
data["countries"] = data.Hotel_Address.apply(lambda x:x.split(' ')[-1])
data.drop(['Additional_Number_of_Scoring', 'Review_Date','Reviewer_Nationality','Negative_Review', 'Review_Total_Negative_Word_Counts','Total_Number_of_Reviews', 'Positive_Review','Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score','days_since_review', 'lat', 'lng'],1,inplace=True)
def impute(column):
  column = column[0]
  if(type(column) !=list):
    return "".join(literal_eval(column))
  else:
    return column
data['Tags'] = data['Tags'].str.lower()
data['countries'] = data['countries'].str.lower()
def recommender(location,description):
  description = description.lower()
  word_tokenize(description)
  stop_words = stopwords.words('english')
  lemm = WordNetLemmatizer()
  filtered = {word for word in description if not word in stop_words}
  filtered_set = set()
  for fs in filtered:
    filtered_set.add(lemm.lemmatize(fs))
  country = data[data['countries']==location.lower()]
  country = country.set_index(np.arange(country.shape[0]))
  list1 = [];list2 = [];cos= [];
  for i in range(country.shape[0]):
    temp_token=word_tokenize(country["Tags"][i])
    temp_set = [word for word in temp_token if not word in stop_words]
    temp2_set = set()
    for s in temp_set:
      temp2_set.add(lemm.lemmatize(s))
    vector = temp2_set.intersection(filtered_set)
    cos.append(len(vector))
  country['similarity']=cos
  country = country.sort_values(by='similarity',ascending=False)
  country.drop_duplicates(subset='Hotel_Name',keep='first',inplace=True)
  country.sort_values('Average_Score',ascending = False,inplace=True)
  country.reset_index(inplace=True)
  return country[["Hotel_Name","Average_Score","Hotel_Address"]].head(10)
recommender('Netherlands','Business trip')
