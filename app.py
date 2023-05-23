# file objective: This file contains all what is required to run predictions on the inputed image(s) in the app
# It is dependent on SAM.py
# function name is "area"
# script owner: Tchako Bryan (PGE 4)
# collaborator: Adetutu (PGE 5)
# script date creation: 23/05/2023


from flask import Flask, request, jsonify, render_template
# other necessary importations

app = Flask(__name__)

# load yolo
# load SAM

@app.route('/')  # your default root page, it will open index.html by default
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # section to render the prediction of sam
    pass

if __name__ == "__main__":
    app.run(debug=True)