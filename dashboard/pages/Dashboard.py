import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import plotly.express as px
from Homepage import add_logo

st.set_page_config(
    page_title="Dashboard Stress Detection on Social Media",
    page_icon=":sparkles:",
    layout="wide"
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
    st.plotly_chart(fig_barchart,use_container_width=True)

## Reddit Stress
def histogram_plot(df,n):
    teks_stress_reddit = df[df['label'] == n]['text']
    teks_stress_reddit = pd.DataFrame({'text': teks_stress_reddit})
    teks_stress_reddit['text_length'] = teks_stress_reddit['text'].apply(lambda x: len(x.split()))
    plt_hist = plt.figure(figsize=(8,8))
    sns.histplot(teks_stress_reddit['text_length'], bins=50, color='blue', kde=True, alpha=0.8)
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    st.plotly_chart(plt_hist,use_container_width=True)

def mean_teks(df,n):
    teks = df[df['label'] == n]['text']
    teks = pd.DataFrame({'text': teks})
    teks['text_length'] = teks['text'].apply(lambda x: len(x.split()))
    mean_length = np.mean(teks['text_length'])
    return round(mean_length)

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
    st.pyplot(fig_world_cloud,use_container_width=True)

def word_count(df, n):
    all_text = ' '.join(df[df['label'] == n]['text'])
    words_stress = all_text.split()
    filter_word_stress = [word for word in words_stress if word.lower() not in stopWords]
    words_stress_freq = Counter(filter_word_stress)
    words_stress_freq = pd.DataFrame(list(words_stress_freq.items()), columns=['Word', 'Frequency'])
    words_stress_freq = words_stress_freq.sort_values(by='Frequency', ascending=False)
    top_10_words_df = words_stress_freq.head(10)
    top_10_words_df = top_10_words_df.iloc[::-1]
    fig = px.bar(
        top_10_words_df,
        x="Frequency",
        y="Word",
        color_discrete_sequence=["#9EE6CF"]
    )
    fig.update_xaxes(title_text="Frequency")
    fig.update_yaxes(title_text="Count")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

## Import The Dataframe
twitter_df = pd.read_csv("dashboard/Data_csv/twitter_new.csv")
reddit_df = pd.read_csv("dashboard/Data_csv/reddit_new.csv")
twitter_df = change_columns(twitter_df)
reddit_df = change_columns(reddit_df)
stopWords = STOPWORDS

## Here Streamlit Code Begin
st.title("Data of Stress Detection from Social Media Articles  :sparkles:")
# select_event = st.sidebar.selectbox('What Social Media do you want to see',['Reddit','Twitter'])

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 2, 0.2, 1, 0.1)
)
row2_spacer1, row2_1, row2_spacer2 = st.columns((0.1, 3.2, 0.1))
#Side bar
# if select_event:
#     select_condition = st.sidebar.selectbox("Which On You Prefer",['Default','Stress','Not Stress'])

row0_1.title("Unveiling the Pulse of Social Stress: A Deep Dive into Twitter and Reddit Data")

with row2_1:
    select_event = st.selectbox('What Social Media do you want to see', ['Reddit', 'Twitter'])
    # if select_event:
    #     select_condition = st.selectbox("Which On You Prefer",['Default','Stress','Not Stress'])
    st.warning(
        """This data based on https://github.com/SenticNet/stress-detection.git 
        This repository contains the datasets for classification of stress from text-based social media articles from Reddit and Twitter, 
        which were created within the paper titled "Stress Detection from Social Media Articles: 
        New Dataset Benchmark and Analytical Study, Thanks To : aryan-r22 """
    )

    st.header("Exploring Stress Levels : **{}** Dataset".format(select_event))


row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)
row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)
row6_space1, row6_1, row6_space2, row6_2, row6_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

if select_event == 'Reddit':
    label_count = define_label_count(reddit_df)
    with row3_1:
        st.metric("Stressed Count", value=label_count.get(1, 0))
        plot_bar_chart(reddit_df)
    with row3_2:
        st.metric("Non Stressed Count", value=label_count.get(0, 0))
    with row4_1:
        st.subheader("Word Distribution of Stress People Based on Word Length")
        mean_stress = mean_teks(reddit_df,1)
        st.metric("The Average Word Length", value=mean_stress)
        histogram_plot(reddit_df, 1)
    with row4_2:
        st.subheader("Word Distribution of Non-Stress People Based on Word Length")
        mean_not_stess = mean_teks(reddit_df, 0)
        st.metric("The Average Word Length", value=mean_not_stess)
        histogram_plot(reddit_df, 0)
    with row5_1:
        st.subheader("Word Cloud of Stress People")
        word_cloud(reddit_df,1)
    with row5_2:
        st.subheader("Word Cloud of Non-Stress People")
        word_cloud(reddit_df, 0)
    with row6_1:
        st.subheader("Most Used Word in Stress Data")
        word_count(reddit_df,1)
    with row6_2:
        st.subheader("Most Used Word in Non-Stress Data")
        word_count(reddit_df, 0)
else:
    label_count = define_label_count(twitter_df)
    with row3_1:
        st.metric("Stressed Count", value=label_count.get(1, 0))
        plot_bar_chart(twitter_df)
    with row3_2:
        st.metric("Non Stressed Count", value=label_count.get(0, 0))
    with row4_1:
        st.subheader("Word Distribution of Stress People Based on Word Length")
        mean_stress = mean_teks(twitter_df, 1)
        st.metric("The Average Word Length", value=mean_stress)
        histogram_plot(twitter_df, 1)
    with row4_2:
        st.subheader("Word Distribution of Non-Stress People Based on Word Length")
        mean_not_stess = mean_teks(twitter_df, 0)
        st.metric("The Average Word Length", value=mean_not_stess)
        histogram_plot(twitter_df, 0)
    with row5_1:
        st.subheader("Word Cloud of Stress People")
        word_cloud(twitter_df, 1)
    with row5_2:
        st.subheader("Word Cloud of Non-Stress People")
        word_cloud(twitter_df, 0)
    with row6_1:
        st.subheader("Most Used Word in Stress Data")
        word_count(twitter_df, 1)
    with row6_2:
        st.subheader("Most Used Word in Non-Stress Data")
        word_count(twitter_df, 0)


# if select_condition == 'Stress' and select_event == 'Reddit':
#     st.markdown("## Bar Chart")
#     plot_bar_chart(reddit_df)
#     st.markdown("## Histogram")
#     histogram_plot(reddit_df,1)
#     st.markdown('## Word Cloud')
#     word_cloud(reddit_df,1)
#     st.markdown('## Word Count')
#     word_count(reddit_df,1)
# elif select_condition == "Not Stress" and select_event == 'Reddit':
#     st.markdown("## Bar Chart")
#     plot_bar_chart(reddit_df)
#     st.markdown("## Histogram")
#     histogram_plot(reddit_df, 0)
#     st.markdown('## Word Cloud')
#     word_cloud(reddit_df, 0)
#     st.markdown('## Word Count')
#     word_count(reddit_df, 0)
# elif select_condition == "Default" and select_event == 'Reddit':
#     st.markdown("## Bar Chart")
#     plot_bar_chart(reddit_df)
#     st.markdown("## Histogram")
#     col1, col2 = st.columns(2)
#     with col1:
#         histogram_plot(reddit_df, 0)
#     with col2:
#         histogram_plot(reddit_df, 1)
#     st.markdown('## Word Cloud')
#     word_cloud(reddit_df, 0)
#     st.markdown('## Word Count')
#     word_count(reddit_df, 0)
# elif select_condition == "Not Stress" and select_event == 'Twitter':
#     st.markdown("## Bar Chart")
#     plot_bar_chart(twitter_df)
#     st.markdown("## Histogram")
#     histogram_plot(twitter_df, 0)
#     st.markdown('## Word Cloud')
#     word_cloud(twitter_df, 0)
#     st.markdown('## Word Count')
#     word_count(twitter_df, 0)
# else:
#     st.markdown("## Bar Chart")
#     plot_bar_chart(twitter_df)
#     st.markdown("## Histogram")
#     histogram_plot(twitter_df, 1)
#     st.markdown('## Word Cloud')
#     word_cloud(twitter_df, 1)
#     st.markdown('## Word Count')
#     word_count(twitter_df, 1)





