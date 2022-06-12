from flask import Flask, render_template, request, jsonify, redirect, url_for, session, make_response, flash

app = Flask(__name__)


@app.route('/')
def index():  
        return render_template('index.html')

@app.route('/output', methods=['POST'])
def output():
    input_text = request.form['input_text']
    ## iterate through series of actions
    label = input_text
    return jsonify({'htmlresponse' : label})

if __name__ == "__main__":  
    app.run(debug=True, port=2000)