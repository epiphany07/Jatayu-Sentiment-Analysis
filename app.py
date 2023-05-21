import base64
import csv
import io
from flask import Flask, request, render_template
from transformers import BertTokenizer, TFBertForSequenceClassification #added
from transformers import BertConfig, BertModel
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #added
model = TFBertForSequenceClassification.from_pretrained("E:\Virtusa\smla\smla\sentimentmodel")

app = Flask(__name__)

@app.route('/')
def my_form():

    return render_template('form.html')
@app.route('/upload', methods=['POST'])
def upload():
    predicted_labels=[]
    
    if 'file' not in request.files:
        return 'No file found!', 400

    file = request.files['file']

    if file.filename == '':
        return 'No file selected!', 400

    df = pd.read_csv(file)

    texts = df['Text'].tolist()
    for text in texts:
        text_split = text.split("\n")
        l=[]
        for i in range(len(text_split)):
            text_final=text_split[i]
            
            tf_batch = tokenizer([text_final],max_length=128, padding=True, truncation=True, return_tensors='tf')
            tf_outputs = model(tf_batch)
            print(tf_outputs)
            tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
            print(tf_predictions)
            label = tf.argmax(tf_predictions, axis=1)
            label = label.numpy()
            l.append([i+1,text_split[i],tf_predictions.numpy()[0,1],tf_predictions.numpy()[0,0]])
            predicted_labels.append( tf_predictions.numpy()[0,1],data=1)
    df['Sentiment'] = predicted_labels
    plt.figure()
    df['Sentiment'].value_counts().plot(kind='bar')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis')
    plt.xticks(rotation=0)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template('result.html', plot_url=plot_url)

@app.route('/', methods=['POST'])
def my_form_post():

    text0 = request.form['text1']
    text_split = text0.split("\n")
    l=[]
    for i in range(len(text_split)):

        text_final=text_split[i]
        
        tf_batch = tokenizer([text_final],max_length=128, padding=True, truncation=True, return_tensors='tf')
        tf_outputs = model(tf_batch)
        print(tf_outputs)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
        print(tf_predictions)
        label = tf.argmax(tf_predictions, axis=1)
        label = label.numpy()
        # l.append([i+1,text_split[i],text_final,tf_predictions.numpy()[0,1],tf_predictions.numpy()[0,0]])
        l.append([i+1,text_split[i],tf_predictions.numpy()[0,1],tf_predictions.numpy()[0,0]])
    return render_template('form.html', final=tf_predictions.numpy()[0,1], data=l)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)



