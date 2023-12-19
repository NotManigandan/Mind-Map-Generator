from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from textinput import text_input
from fileupload import file_upload
from webscrape import web_scrape

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        option = request.form['option']
        try:
            if option == '1':
                result_image = text_input(request.form['input_text'])
            elif option == '2':
                result_image = file_upload(request.files['file'])
            elif option == '3':
                result_image = web_scrape(request.form['website_link'])
            else:
                return render_template('index.html', error='Invalid option selected')

            if result_image is None:
                return render_template('index.html', error='Error processing request')

            return redirect(url_for('result'))

        except Exception as e:
            print(f"Error processing request: {e}")
            return render_template('index.html', error='Error processing request')

    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
