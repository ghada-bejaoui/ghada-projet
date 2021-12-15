from flask import Flask
from flask import render_template
import os

PEOPLE_FOLDER = os.path.join('static', 'people_photo')

app=Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDE



@app.route('/upload')
def upload_files():
   full_filename = os.path.join(app.config['UPLOAD_FOLDER'], '/static/bg.png')
   return render_template("form.html", user_image = full_filename)


		
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090) 
