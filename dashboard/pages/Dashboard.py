import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import warnings
import os
from Homepage import add_logo

st.set_page_config(
    page_title="Dashboard Stress Detection on Social Media",
    page_icon=":sparkles:",
)

sns.set(style='dark')
add_logo()

## Function All Purpose
def change_columns(df):
    new_column_names = {df.columns[0]: 'text'}
    df.rename(columns=new_column_names, inplace=True)
    return df
def define_label(df):
    label_name = df['label']
    label_name = label_name.replace({1: 'Stress', 0: 'Not Stress'})
    label_name = label_name.unique()
    return label_name

def define_label_count(df):
    label_count = df['label'].value_counts()
    return label_count

## For Reddit
def plot_bar_chart(df):
    label_name = define_label(df)
    label_counts = define_label_count(df)
    fig_barchart = plt.figure(figsize=(8, 6))
    plt.bar(label_name, label_counts.values)
    plt.xlabel('Label')
    plt.ylabel('Sum')
    st.pyplot(fig_barchart)

## Reddit Stress
def histogram_plot(df,n):
    teks_stress_reddit = df[df['label'] == n]['text']
    teks_stress_reddit = pd.DataFrame({'text': teks_stress_reddit})
    teks_stress_reddit['text_length'] = teks_stress_reddit['text'].apply(lambda x: len(x.split()))
    plt_hist = plt.figure(figsize=(8,8))
    sns.histplot(teks_stress_reddit['text_length'], bins=50, color='blue', kde=True, alpha=0.8)
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    st.pyplot(plt_hist)

def word_cloud(df,n):
    all_text_stress_redd = ' '.join(df[df['label'] == n]['text'])
    wordcloud_stress_redd = WordCloud(
        background_color='black',
        height=1080,
        width=1920,
        stopwords=stopWords
    )
    wordcloud_stress_redd.generate(all_text_stress_redd)
    fig_world_cloud = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_stress_redd, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig_world_cloud)

def word_count(df,n):
    all_text_stress = ' '.join(df[df['label'] == n]['text'])
    words_stress = all_text_stress.split()
    filter_word_stress = [word for word in words_stress if word.lower() not in stopWords]
    words_stress_freq= Counter(filter_word_stress)
    words_stress_freq = pd.DataFrame(list(words_stress_freq.items()), columns=['Word', 'Frequency'])
    words_stress_freq = words_stress_freq.sort_values(by='Frequency', ascending=False)
    colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    fig_plot = plt.figure(figsize=(8, 6))
    sns.barplot(x="Frequency", y="Word", data=words_stress_freq.head(10), palette=colors)
    plt.ylabel(None)
    plt.xlabel("Frequency", fontsize=15)
    plt.tick_params(axis='y', labelsize=12)
    st.pyplot(fig_plot)


## Import The Dataframe
twitter_df = pd.read_csv("../dashboard/Data_csv/twitter_new.csv")
reddit_df = pd.read_csv("../dashboard/Data_csv/reddit_new.csv")
twitter_df = change_columns(twitter_df)
reddit_df = change_columns(reddit_df)
stopWords = STOPWORDS

## Here Streamlit Code Begin
st.title("Stress Detection on Social Media :sparkles:")
select_event = st.sidebar.selectbox('What Social Media do you want to see',['Reddit','Twitter'])

if select_event:
    select_condition = st.sidebar.selectbox("Which On You Prefer",['Default','Stress','Not Stress'])

if select_condition == 'Stress' and select_event == 'Reddit':
    st.markdown("## Bar Chart")
    plot_bar_chart(reddit_df)
    st.markdown("## Histogram")
    histogram_plot(reddit_df,1)
    st.markdown('## Word Cloud')
    word_cloud(reddit_df,1)
    st.markdown('## Word Count')
    word_count(reddit_df,1)
elif select_condition == "Not Stress" and select_event == 'Reddit':
    st.markdown("## Bar Chart")
    plot_bar_chart(reddit_df)
    st.markdown("## Histogram")
    histogram_plot(reddit_df, 0)
    st.markdown('## Word Cloud')
    word_cloud(reddit_df, 0)
    st.markdown('## Word Count')
    word_count(reddit_df, 0)
elif select_condition == "Default" and select_event == 'Reddit':
    st.markdown("## Bar Chart")
    plot_bar_chart(reddit_df)
    st.markdown("## Histogram")
    col1, col2 = st.columns(2)
    with col1:
        histogram_plot(reddit_df, 0)
    with col2:
        histogram_plot(reddit_df, 1)
    st.markdown('## Word Cloud')
    word_cloud(reddit_df, 0)
    st.markdown('## Word Count')
    word_count(reddit_df, 0)
elif select_condition == "Not Stress" and select_event == 'Twitter':
    st.markdown("## Bar Chart")
    plot_bar_chart(twitter_df)
    st.markdown("## Histogram")
    histogram_plot(twitter_df, 0)
    st.markdown('## Word Cloud')
    word_cloud(twitter_df, 0)
    st.markdown('## Word Count')
    word_count(twitter_df, 0)
else:
    st.markdown("## Bar Chart")
    plot_bar_chart(twitter_df)
    st.markdown("## Histogram")
    histogram_plot(twitter_df, 0)
    st.markdown('## Word Cloud')
    word_cloud(twitter_df, 0)
    st.markdown('## Word Count')
    word_count(twitter_df, 0)





