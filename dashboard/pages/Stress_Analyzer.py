import streamlit as st
from Homepage import add_logo
from transformers import TFBertModel
from transformers import BertTokenizerFast
import tensorflow as tf
import numpy as np

###
custom_objects = {'TFBertModel': TFBertModel}

def analyze_bert(input_text):
    loaded_model = tf.keras.models.load_model('dashboard/Model/model_bert.h5',
                                            custom_objects=custom_objects)
    tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-uncased')
    text_input = tokenizer_bert.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        return_attention_mask=True
    )

    input_ids = text_input['input_ids']
    attention_mask = text_input['attention_mask']
    # Assuming input_ids_np and attention_mask_np are NumPy arrays
    input_ids_np = np.array(input_ids)
    attention_mask_np = np.array(attention_mask)

    # Add a batch dimension
    input_ids_np = np.expand_dims(input_ids_np, axis=0)
    attention_mask_np = np.expand_dims(attention_mask_np, axis=0)

    input_ids_tensor = tf.constant(input_ids_np)
    attention_mask_tensor = tf.constant(attention_mask_np)

    with tf.device('/CPU:0'):
        predicted = loaded_model([input_ids_tensor, attention_mask_tensor])

    # Convert the result to a NumPy array if needed
    predicted_np = predicted.numpy()

    predicted_labels = np.argmax(predicted, axis=1)

    if predicted_labels[0] == 1:
        return "You Are Stress, Get Some Rest :sos:"
    else:
        return "You Are Good, Keep it Up :sparkles:"


st.set_page_config(
    page_title="Stress Analyzer on Social Media",
    page_icon=":sparkles:",
    layout="wide"
)
add_logo()
st.title("Welcome to Stress Analyzer")

select_model = st.sidebar.selectbox('Which Model Do You Want To Use',
                    (
                        'BERT',
                        'RoBERTa'
                    ))

user_input = st.text_input("Enter Your Text Here :", "")
analyze_button = st.button("Analyze")

if select_model == 'BERT' and analyze_button:
    # Add a spinner to indicate loading
    with st.spinner("Analyzing..."):
        # Perform analysis
        hasil = analyze_bert(user_input)

    # Display the result after analysis
    st.subheader(hasil)


