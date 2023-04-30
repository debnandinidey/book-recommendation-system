import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv('C:/Users/Nandini/book-recommender/data/Books.csv')
users = pd.read_csv('C:/Users/Nandini/book-recommender/data/Users.csv')
ratings = pd.read_csv('C:/Users/Nandini/book-recommender/data/Ratings.csv')
# Datafile Exploration
books.head(3)
books.info()
books.describe()
users.head(3)
users.info()
users.describe()
print(books.shape)
print(ratings.shape)
print(users.shape)

#checking the missing values
books.isnull().sum()
features_with_na=[features for features in books.columns if books[features].isnull().sum()>=1]

#dropping all the null values
books = books.dropna()

#checking for duplicate values
books.duplicated().sum()
# Lets remane few columns name from books
books.rename(columns={"Book-Title":"title",
                      "Book-Author":"author",
                     "Year-Of-Publication":"year",
                     "Publisher":"publisher",
                     "Image-URL-L":"image_url_l",
                     "Image-URL-S":"image_url_s",
                     "Image-URL-M":"image_url_m"},inplace=True)
ratings.rename(columns={"User-ID":"user_id",
                      "Book-Rating":"rating"},inplace=True)
books = books.dropna()
ratings_with_bookname=ratings.merge(books, on= "ISBN")
#on ISBN because ISBN is common in both the columns
ratings_with_bookname.groupby('title').count()['rating'].reset_index()
num_rating_df = ratings_with_bookname.groupby('title').count()['rating'].reset_index()
num_rating_df.rename(columns={'rating':'num_ratings'},inplace=True)
avg_rating_df = ratings_with_bookname.groupby('title').mean()['rating'].reset_index()
avg_rating_df.rename(columns={'rating':'avg_rating'},inplace=True)
popularity_df = num_rating_df.merge(avg_rating_df,on='title')
popularity_df = popularity_df[popularity_df['num_ratings']>=250]
popularity_df = popularity_df[popularity_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)
popularity_df = popularity_df.merge(books,on='title').drop_duplicates('title')[['title','publisher','author','image_url_m','num_ratings','avg_rating']]
#userid and their respective number of ratings
x=ratings_with_bookname.groupby('user_id').count()['rating']>200
x[x]
x=ratings_with_bookname.groupby('user_id').count()['rating']>200
x[x].index
valuable_users=x[x].index
ratings_with_bookname['user_id'].isin(valuable_users)
filtered_rating=ratings_with_bookname[ratings_with_bookname['user_id'].isin(valuable_users)]
filtered_rating.groupby('title').count()['rating']

#books which has more than 50 rating
y=filtered_rating.groupby('title').count()['rating']>=50
y[y]
valuable_books=y[y].index
final_ratings= filtered_rating[filtered_rating['title'].isin(valuable_books)]
pivot=final_ratings.pivot_table(index='title',columns='user_id',values='rating')
pivot.fillna(0,inplace=True)

similarity_scores = cosine_similarity(pivot)
list(enumerate(similarity_scores[0]))
#similarity_score[0] means the first index which is the first book(1989) and its respective similarity score with each of the other books.enumerate function just gives the value in the form of a list along with its respective index.

def recommend(book_name):
    # index fetch
    index = np.where(pivot.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
    # enumerate gives the value in the form of a list along with the indexs
    # reverse=True means the similarity score will be fetched in descending order
    # key=lambda x:x[1] means the sorting will be based on the similarity score, not the indexes

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['title'] == pivot.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('title')['title'].values))
        item.extend(list(temp_df.drop_duplicates('title')['publisher'].values))
        item.extend(list(temp_df.drop_duplicates('title')['author'].values))
        item.extend(list(temp_df.drop_duplicates('title')['image_url_m'].values))

        data.append(item)

    return data
import pickle
pickle.dump(popularity_df,open('popularity.pkl','wb'))
books.drop_duplicates('title')
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))
pickle.dump(pivot,open('pivot.pkl','wb'))






