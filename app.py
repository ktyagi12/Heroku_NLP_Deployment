from flask import Flask, request, render_template
import pickle

# creating a Flask app
my_app = Flask(__name__)

# loading the model
nlp_model = pickle.load(open('nlp_model.pkl','rb'))

# loading the Count Vectoriser
count_vec = pickle.load(open('countVec.pkl','rb'))

@my_app.route('/')
def home():
    return render_template('home.html')

@my_app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        output_msg = ''

        message = request.form['message']
        data = [message]

        vec = count_vec.transform(data).toarray()
        predict_target = nlp_model.predict(vec)

        if predict_target == 1:
            output_msg = 'Oh Thank God! You are safe. It is a ham'

        elif predict_target == 0:
            output_msg = 'Oops! You have received a spam message.'

    return render_template('home.html', prediction_value=output_msg)


if __name__=='__main__':
    my_app.run(debug=True)
