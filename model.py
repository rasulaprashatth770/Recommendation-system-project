# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 


st.title('Model Deployment: User-Based Model')

st.sidebar.header('User Input Parameters')

def user_input_features():
    UserID = st.sidebar.number_input("Insert UserID")
    return UserID


u = user_input_features()
st.subheader('User Input parameters')
st.write('UserID',u)

agg_ratings_GT100 = pd.read_csv('grouped_data_clean.csv')
agg_ratings_GT100.drop('Unnamed: 0',axis=1,inplace=True)
df = pd.read_csv('Data_Clean1.csv')

# Merge data
df_GT100 = pd.merge(df, agg_ratings_GT100[['booktitle']], on='booktitle', how='inner')
df_GT100.dropna(inplace=True)
matrix = df_GT100.pivot_table(index='UserID', columns='booktitle', values='Rating')
st.write('User vs Booktitle Matrix',matrix)

# Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis='rows')

# User similarity matrix using Pearson correlation
user_similarity = matrix_norm.T.corr()

# Pick a user ID
picked_userid = u

# Remove picked user ID from the candidate list
user_similarity.drop(index=picked_userid, inplace=True)

# User similarity threashold
user_similarity_threshold = 0.3

# Get top n similar users
similar_users = user_similarity[user_similarity[picked_userid]>user_similarity_threshold][picked_userid].sort_values(ascending=False)

st.write('Similar users to selected user are:',similar_users)

# Books that the target user has read
picked_userid_read = matrix_norm[matrix_norm.index == picked_userid].dropna(axis=1, how='all')
st.write('Books picked user read:',picked_userid_read)

# Books that similar users read. Remove books that none of the similar users have read.
similar_user_books = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1,how='all')

# Remove the read books from the book list
similar_user_books.drop(picked_userid_read.columns,axis=1, inplace=True, errors='ignore')
st.write('Books that similar users read:',similar_user_books)

# A dictionary to store item scores
item_score = {}

# Loop through items
for i in similar_user_books.columns:
    # Get the ratings for book i
    Book_rating = similar_user_books[i]
    # Create a variable to store the score
    total = 0
    # Create a variable to store the number of scores
    count = 0
    # Loop through similar users
    for u in similar_users.index:
        # If the book has rating
        if pd.isna(Book_rating[u]) == False:
            # Score is the sum of user similarity score multiply by the book rating
            score = similar_users[u] * Book_rating[u]
            # Add the score to the total score for the book so far
            total += score
            # Add 1 to the count
            count += 1
    # Get the average score for the item
    item_score[i] = total / count

# Convert dictionary to pandas dataframe
item_score = pd.DataFrame(item_score.items(), columns=['book', 'book_score'])

# Sort the books by score
ranked_item_score = item_score.sort_values(by='book_score', ascending=False)

# Select top m books
m = 10


# Average rating for the picked user
avg_rating = matrix[matrix.index == picked_userid].T.mean()[picked_userid]
st.write('Average rating given by picked user',avg_rating)

# Calcuate the predicted rating
ranked_item_score['predicted_rating'] = ranked_item_score['book_score'] + avg_rating

# Take a look at the data
st.write('Recommended Books are:',ranked_item_score.head(m))