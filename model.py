
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


st.title('User-Based Recommendation System')
st.sidebar.header('User Input Parameters')

agg_ratings_GT100 = pd.read_csv('grouped_data_clean.csv')
agg_ratings_GT100.drop('Unnamed: 0',axis=1,inplace=True)
book = pd.read_csv('Books.csv',error_bad_lines=False,encoding="latin-1")
df = pd.read_csv('Data_Clean1.csv')
df1 = pd.read_csv('df2_mrg.csv')
popular_df = pd.read_csv('popular_df.csv')

popular_df = popular_df.sort_values('avg_ratings',ascending=False)
popular_df.drop('Unnamed: 0', axis=1, inplace=True)
popular_df.drop('num_ratings', axis=1, inplace=True)

# Converting links to html tags
def img_t (path):
    return '"' + path + '"'

popular_df['book'] = popular_df['Title']
popular_df.drop('Title',axis=1,inplace=True)
popular_df.sort_values('avg_ratings',ascending=False,inplace=True)


def user_input_features():
    UserID = st.sidebar.number_input("Insert UserID")
    return UserID


u = user_input_features()



df_GT100 = pd.merge(df, agg_ratings_GT100[['booktitle']], on='booktitle', how='inner')
df_GT100.dropna(inplace=True)

# Create user-item matrix
matrix = df_GT100.pivot_table(index='UserID', columns='booktitle', values='Rating')

# Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis='rows')

# User similarity matrix using Pearson correlation
user_similarity = matrix_norm.T.corr()

try:

    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""<p class="big-font">Books recommended for picked user are</p>""", unsafe_allow_html=True)

    picked_userid = u

    # Remove picked user ID from the candidate list
    user_similarity.drop(index=picked_userid, inplace=True)

    # User similarity threashold
    user_similarity_threshold = 0.3

    # Get top n similar users
    similar_users = user_similarity[user_similarity[picked_userid] > user_similarity_threshold][
        picked_userid].sort_values(ascending=False)

    # Books that the target user has read
    picked_userid_read = matrix_norm[matrix_norm.index == picked_userid].dropna(axis=1, how='all')

    # Books that similar users read. Remove books that none of the similar users have read.
    similar_user_books = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')

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
        for f in similar_users.index:
            # If the book has rating
            if pd.isna(Book_rating[f]) == False:
                # Score is the sum of user similarity score multiply by the book rating
                score = similar_users[f] * Book_rating[f]
                # Add the score to the total score for the book so far
                total += score
                # Add 1 to the count
                count += 1
        # Get the average score for the item
        item_score[i] = total / count

    # Convert dictionary to pandas dataframe
    item_score = pd.DataFrame(item_score.items(), columns=['book', 'book_score'])

    # Sort the books by score
    ranked_item_score = item_score

    # Select top m books
    m = 10

    # Average rating for the picked user
    avg_rating = matrix[matrix.index == picked_userid].T.mean()[picked_userid]

    # Calcuate the predicted rating
    ranked_item_score['predicted_rating'] = ranked_item_score['book_score'] + avg_rating

    z = ranked_item_score.sort_values('predicted_rating')

    x = pd.DataFrame({'book': z['book']})
    y = pd.DataFrame({'book': popular_df['book']})

    recommend = pd.concat([x, y])
    recommend.reset_index(drop=True, inplace=True)

    recommend1 = pd.merge(recommend, popular_df, on='book', how='left')
    # recommend1['Image_URL_M']=pd.DataFrame(img_t(recommend1['Image_URL_M']))
    # book['Image-URL-M']=pd.DataFrame(img_t(book['Image-URL-M']))

    # Load the CSV file with NaN values
    df_with_nan = pd.DataFrame(recommend1)
    df_with_nan.rename(columns={'book': 'booktitle', 'Author': 'bookAuthor'}, inplace=True)
    # df_with_nan.drop('Unnamed: 0',axis=1,inplace=True)

    # Load the CSV file with complete values
    df_complete_values = pd.read_csv('df2_mrg.csv')

    # Identify the common column in both DataFrames
    common_column = 'booktitle'

    # Merge the two DataFrames on the common column
    merged_df = pd.merge(df_with_nan, df_complete_values[[common_column, 'bookAuthor']], on=common_column, how='left')

    # Replace NaN values in the first DataFrame with corresponding values from the second DataFrame
    merged_df['bookAuthor_x'] = merged_df['bookAuthor_x'].fillna(merged_df['bookAuthor_y'])

    # Drop the 'other_column' if no longer needed
    merged_df.drop('bookAuthor_y', axis=1, inplace=True)

    merged_df.drop_duplicates(inplace=True)
    merged_df.drop(['avg_ratings', 'Image_URL_M'], axis=1, inplace=True)
    merged_df.dropna(inplace=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write('Book Title: ', merged_df.iloc[0]['booktitle'])
        st.write('Author: ', merged_df.iloc[0]['bookAuthor_x'])
    with c3:
        s = merged_df.iloc[0]['booktitle']
        a = merged_df.iloc[0]['bookAuthor_x']
        t = pd.DataFrame(df1.loc[(df1['booktitle'] == s) & (df1['bookAuthor'] == a)])
        i = t.iloc[0]['imageUrlmL']
        st.image(i, use_column_width=True)
    st.divider()

    c4, c5, c6 = st.columns(3)

    with c4:
        st.write('Book Title: ', merged_df.iloc[1]['booktitle'])
        st.write('Author: ', merged_df.iloc[1]['bookAuthor_x'])
    with c6:
        s1 = merged_df.iloc[1]['booktitle']
        a1 = merged_df.iloc[1]['bookAuthor_x']
        t1 = pd.DataFrame(df1.loc[(df1['booktitle'] == s1) & (df1['bookAuthor'] == a1)])
        i1 = t1.iloc[1]['imageUrlmL']
        st.image(i1, use_column_width=True)
    st.divider()

    c7, c8, c9 = st.columns(3)

    with c7:
        st.write('Book Title: ', merged_df.iloc[2]['booktitle'])
        st.write('Author: ', merged_df.iloc[2]['bookAuthor_x'])
    with c9:
        s2 = merged_df.iloc[2]['booktitle']
        a2 = merged_df.iloc[2]['bookAuthor_x']
        t2 = pd.DataFrame(df1.loc[(df1['booktitle'] == s2) & (df1['bookAuthor'] == a2)])
        i2 = t2.iloc[2]['imageUrlmL']
        st.image(i2, use_column_width=True)
    st.divider()

    c10, c11, c12 = st.columns(3)

    with c10:
        st.write('Book Title: ', merged_df.iloc[3]['booktitle'])
        st.write('Author: ', merged_df.iloc[3]['bookAuthor_x'])
    with c12:
        s3 = merged_df.iloc[3]['booktitle']
        a3 = merged_df.iloc[3]['bookAuthor_x']
        t3 = pd.DataFrame(df1.loc[(df1['booktitle'] == s3) & (df1['bookAuthor'] == a3)])
        i3 = t3.iloc[3]['imageUrlmL']
        st.image(i3, use_column_width=True)
    st.divider()

    c13, c14, c15 = st.columns(3)

    with c13:
        st.write('Book Title: ', merged_df.iloc[4]['booktitle'])
        st.write('Author: ', merged_df.iloc[4]['bookAuthor_x'])
    with c15:
        s4 = merged_df.iloc[4]['booktitle']
        a4 = merged_df.iloc[4]['bookAuthor_x']
        t4 = pd.DataFrame(df1.loc[(df1['booktitle'] == s4) & (df1['bookAuthor'] == a4)])
        i4 = t4.iloc[4]['imageUrlmL']
        st.image(i4, use_column_width=True)

except:

    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""<p class="big-font">Haven't started reading yet. Try These:</p>""", unsafe_allow_html=True)

    c16,c17,c18=st.columns(3)

    with c16:
        st.write('Book Title: ',popular_df.iloc[0]['book'])
        st.write('Author: ',popular_df.iloc[0]['Author'])

    with c18:
        i5 = popular_df.iloc[0]['Image_URL_M']
        st.image(i5,use_column_width='always')
    st.divider()

    c19, c20, c21 = st.columns(3)

    with c19:
        st.write('Book Title: ', popular_df.iloc[1]['book'])
        st.write('Author: ', popular_df.iloc[1]['Author'])

    with c21:
        i6 = popular_df.iloc[1]['Image_URL_M']
        st.image(i6, use_column_width='always')
    st.divider()

    c22, c23, c24 = st.columns(3)

    with c22:
        st.write('Book Title: ', popular_df.iloc[8]['book'])
        st.write('Author: ', popular_df.iloc[8]['Author'])

    with c24:
        i8 = popular_df.iloc[8]['Image_URL_M']
        st.image(i8, use_column_width='always')
    st.divider()

    c25, c26, c27 = st.columns(3)

    with c25:
        st.write('Book Title: ', popular_df.iloc[5]['book'])
        st.write('Author: ', popular_df.iloc[5]['Author'])

    with c27:
        i9 = popular_df.iloc[5]['Image_URL_M']
        st.image(i9, use_column_width='always')
    st.divider()

    c28, c29, c30 = st.columns(3)

    with c28:
        st.write('Book Title: ', popular_df.iloc[4]['book'])
        st.write('Author: ', popular_df.iloc[4]['Author'])

    with c30:
        i10 = popular_df.iloc[4]['Image_URL_M']
        st.image(i10, use_column_width='always')


