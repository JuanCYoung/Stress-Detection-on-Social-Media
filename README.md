# Stress-Detection-on-Social-Media

## Streamlit Cloud Link
https://stress-detection-on-social-media-wjkkamhvdpkzpzguah4zgz.streamlit.app/

## Description
This repository contains a data science project and also dashboard of Stress Classification Models utilizing BERT and RoBERTa for analyzing data sourced from Reddit and Twitter. In addition to the pre-trained models, it also offers Exploratory Data Analysis (EDA) tools to gain insights into patterns and trends related to stress among individuals in the provided social media data. Users can leverage the dashboard to explore, visualize, and comprehend the results of stress classification, fostering a deeper understanding of stress-related discourse in online platforms.

## Repository Structure

- **Stress_detection_notebook.ipynb**: This file is a notebook used to train the model and also analyzing the dataset
- **/dashboard/model**: contains BERT and RoBERTa model
- **/dashboard/Data_csv**: contains the used dataset
- **/dashboard/Homepage.py**: contains Homepage dashboard
- **/dashboard/pages/Dashboard.py**: contains EDA dashboard
- **/dashboard/pages/Stress_Analyzer.py**: contains page where you can test our model

## Note

This repository consists of two branches. The main branch is where the Streamlit cloud code is hosted (No Stress Analyzer). If you wish to utilize all the features, I recommend cloning and running the code locally from the "Local" branch. However, please note that the "Local" branch's "dashboard/Model" contains only one model due to the limitations of GIT LFS. To access the remaining models, I recommend downloading them from the following link: https://drive.google.com/drive/u/0/folders/10n5BVkS-vinlMZxGgfUEYSHI380HqVQ6

## How To Install

1. Clone repository 
   (Local Branch)
   ```shell 
   git clone -b PC https://github.com/JuanCYoung/Stress-Detection-on-Social-Media.git
   ```
   or
   (Main Branch)
    ```shell
   git clone https://github.com/JuanCYoung/Stress-Detection-on-Social-Media.git
   ```

2. Install the necessary packages by running the following commands:

    ```shell
    pip install streamlit
    pip install -r requirements.txt
    ```
## Usage
1. Navigate to the local directory by running the following command:

    ```shell
    cd dashboard/streamlit run Homepage.py
    ```
## Our Feature

![](https://github.com/JuanCYoung/Stress-Detection-on-Social-Media/blob/main/gif/Homepage.gif)
![](https://github.com/JuanCYoung/Stress-Detection-on-Social-Media/blob/main/gif/EDA.gif)
![](https://github.com/JuanCYoung/Stress-Detection-on-Social-Media/blob/main/gif/Stress.gif)
![](https://github.com/JuanCYoung/Stress-Detection-on-Social-Media/blob/main/gif/NonStress.gif)
