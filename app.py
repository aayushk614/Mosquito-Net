from flask import Flask, request, render_template
from inference import get_result
import os
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        img = file.read()
        predicted , result = get_result(image_bytes=img)
        return render_template('result.html', result=result, predicted=predicted)

if __name__ == '__main__':
 app.run(debug=True,port=os.getenv('PORT',5000))

