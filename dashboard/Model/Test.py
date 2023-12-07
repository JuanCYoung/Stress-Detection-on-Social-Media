from transformers import TFBertModel
from transformers import BertTokenizerFast
import tensorflow as tf
import numpy as np
from Homepage import add_logo

# # Define custom_objects dictionary to map the custom layers
custom_objects = {'TFBertModel': TFBertModel}
#
# # Load the saved model with custom_objects
loaded_model = tf.keras.models.load_model('D:/Github/Stress-Detection-on-Social-Media/Model/model_bert.h5', custom_objects=custom_objects)
StressText = "I want to see world Burn, I hate you all"
tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-uncased')
test_input = tokenizer_bert.encode_plus(
            StressText,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_attention_mask=True
        )
input_ids = test_input['input_ids']
attention_mask = test_input['attention_mask']
# Assuming input_ids_np and attention_mask_np are NumPy arrays
input_ids_np = np.array(input_ids)
attention_mask_np = np.array(attention_mask)

# Add a batch dimension
input_ids_np = np.expand_dims(input_ids_np, axis=0)
attention_mask_np = np.expand_dims(attention_mask_np, axis=0)

# Convert NumPy arrays to TensorFlow tensors
input_ids_tensor = tf.constant(input_ids_np)
attention_mask_tensor = tf.constant(attention_mask_np)

# Predict outside of tf.function
with tf.device('/CPU:0'):  # Workaround to run outside of GPU
    predicted = loaded_model([input_ids_tensor, attention_mask_tensor])

# Convert the result to a NumPy array if needed
predicted_np = predicted.numpy()

# Print or use the result as needed
print(predicted_np)


predicted_labels = np.argmax(predicted, axis=1)

print(predicted_labels)


