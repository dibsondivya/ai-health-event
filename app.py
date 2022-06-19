from flask import Flask, render_template, request, jsonify, redirect, url_for, session, make_response, flash

from bertpredict import convert_label
from distilbertpredict import predict

app = Flask(__name__)


@app.route('/')
def index():  
        return render_template('index.html')

@app.route('/output', methods=['POST'])
def output():
    input_text = request.form['input_text']
    ## iterate through series of actions
    label = input_text
    predicted_label = predict(label)
    predicted_label = convert_label(predicted_label)
    out = f'The tweet is a(n) {predicted_label.upper()}.'
    return jsonify({'htmlresponse' : out})

if __name__ == "__main__":  
    app.run(debug=True, port=2000)