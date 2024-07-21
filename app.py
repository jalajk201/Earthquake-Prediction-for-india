from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/home')
def home2():
    return render_template('homepage.html')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    data1 = int(float(request.form['a']))
    data2 = int(float(request.form['b']))
    data3 = int(float(request.form['c']))
    print(data1, data2, data3)
    arr = np.array([[data1, data2, data3]])
    output = model.predict(arr)

    def to_str(var):
        return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

   
    rounded_output = [int(round(pred)) for pred in output]

    
    if all(pred < 4 for pred in rounded_output):
        return render_template('prediction.html', p=to_str(rounded_output), q='No')
    elif all(4 <= pred < 6 for pred in rounded_output):
        return render_template('prediction.html', p=to_str(rounded_output), q='Low')
    elif all(6 <= pred < 8 for pred in rounded_output):
        return render_template('prediction.html', p=to_str(rounded_output), q='Moderate')
    elif all(8 <= pred < 9 for pred in rounded_output):
        return render_template('prediction.html', p=to_str(rounded_output), q='High')
    elif all(pred >= 9 for pred in rounded_output):
        return render_template('prediction.html', p=to_str(rounded_output), q='Very High')
    else:
        return render_template('prediction.html', p='N.A.', q='Undefined')

if __name__ == "__main__":
    app.run(debug=True)
