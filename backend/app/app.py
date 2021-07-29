<<<<<<< HEAD
from flask import Flask, request, send_file, jsonify, url_for, redirect, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship
import pymysql
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from skimage import io
import cv2
from werkzeug.utils import secure_filename
import background_remove.process as process
from google.cloud import storage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
upload_folder = './static/'

def datetime():
    import datetime
    now = datetime.datetime.now()
    return now.strftime('%Y%m%d%H%M%S')

def upload_blob(source_file_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket('linist_1')
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(source_file_name)


def load_img(image_location):
    img = io.imread(image_location)
    img = cv2.resize(img[:, :, 0:3], (256, 256), interpolation=cv2.INTER_AREA)
    return img


def save_img(img, image_name):
    os.makedirs(upload_folder, exist_ok=True)  # 폴더가 없을 경우 생성
    image_location = os.path.join(upload_folder + secure_filename(image_name))
    cv2.imwrite(image_location, img)
    return image_location


def removeAllFile(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)


def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    ret, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:tkdlzh1gh@localhost/linist_db'
app.config['SECRET_KEY'] = "random string"

db = SQLAlchemy(app)

class users(db.Model):
   id = db.Column('id', db.Integer, primary_key = True)
   name = db.Column(db.String(100))
   email = db.Column(db.String(50), primary_key = True)
   pw = db.Column(db.String(200)) 
   
   def __init__(self, name, email, pw):
       self.name = name
       self.email = email
       self.pw = pw

class images(db.Model):
    Index = db.Column('Index', db.Integer, primary_key = True)
    #users_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    img_url = db.Column(db.String(100))
    date = db.Column(db.String(100)) 

    #users = relationship("users")
   
    def __init__(self, img_url, date):
        #self.users_id = users_id
        self.img_url = img_url
        self.date = date

def connectdb(url, date):
    new_image = images(url, date)
    db.session.add(new_image)
    db.session.commit()

@app.route('/')
def index():
    return '내 추억의 연장선 On the Line'


@app.route('/img_trans', methods=['POST', 'GET'])
def img_trans():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            timedata = datetime()
            image_location = os.path.join(
                upload_folder + secure_filename(image_file.filename))
            image_file.save(image_location)
            upload_blob(image_location, timedata + 'image.png')

            back_removed = process.process(image_location)
            back_removed_name = 'back_removed_' + 'test.png'
            back_location = upload_folder + back_removed_name
            back_removed.save(back_location)
            upload_blob(back_location, timedata + 'back_removed.png')

            img = cv2.imread(back_location)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)

            total_color = 9

            img = color_quantization(img, total_color)

            blurred = cv2.bilateralFilter(
                img, d=7, sigmaColor=250, sigmaSpace=250)
            linist = cv2.bitwise_and(blurred, blurred, mask=edges)

            linist_name = upload_folder + 'linist_test.png'
            cv2.imwrite(linist_name, linist)
            linist = process.process(linist_name)
            linist.save(linist_name)
            upload_blob(linist_name, timedata + 'linist.png')

            linist_url = 'https://storage.googleapis.com/linist_1/' + timedata + 'linist.png'
            removeAllFile(upload_folder)

            return linist_url, connectdb(linist_url, timedata)

        else:
            return jsonify({'Result': 'Fail'})

    else:
        return '내 추억의 연장선 On the Line'


@app.route('/background', methods=['POST', 'GET'])
def background():
    if request.method == 'POST':
        if request.files.get('image') and request.files.get('backImage'):
            timedata = datetime()
            image = request.files['image']
            backImage = request.files['backImage']

            image_name = secure_filename(image.filename)
            image_location = os.path.join(
                upload_folder + image_name)
            image.save(image_location)
            image = Image.open(image_location)
            image.convert("RGBA")
            imsize = image.size

            back_location = os.path.join(
                upload_folder + secure_filename(backImage.filename))
            backImage.save(back_location)
            backImage = Image.open(back_location)
            backImage = backImage.resize(imsize)

            datas = image.getdata()
            datas_back = backImage.getdata()

            finalData = list()

            for i in range(len(datas)):
                if datas[i][3] == 0:
                    finalData.append(datas_back[i])
                else:
                    finalData.append(datas[i])

            image.putdata(finalData)
            image.save(upload_folder + 'final.png')

            upload_blob(upload_folder + 'final.png', timedata + 'final.png')

            final_url = 'https://storage.googleapis.com/linist_1/' + timedata + 'final.png'

            removeAllFile(upload_folder)

            return final_url, connectdb(final_url, timedata)

        else:
            return jsonify({'Result': 'Fail'})
    else:
        return '내 추억의 연장선 On the Line'


if __name__ == '__main__':
    db.create_all()
=======
from flask import Flask, request, send_file, jsonify, url_for, redirect, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from skimage import io
import cv2
from werkzeug.utils import secure_filename
import background_remove.process as process
from google.cloud import storage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
upload_folder = './static/'

def datetime():
    import datetime
    now = datetime.datetime.now()
    return now.strftime('%Y%m%d%H%M%S')

def upload_blob(source_file_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket('linist_1')
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(source_file_name)


def load_img(image_location):
    img = io.imread(image_location)
    img = cv2.resize(img[:, :, 0:3], (256, 256), interpolation=cv2.INTER_AREA)
    return img


def save_img(img, image_name):
    os.makedirs(upload_folder, exist_ok=True)  # 폴더가 없을 경우 생성
    image_location = os.path.join(upload_folder + secure_filename(image_name))
    cv2.imwrite(image_location, img)
    return image_location


def removeAllFile(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)


def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    ret, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:ewha1886@localhost/linist_db'
app.config['SECRET_KEY'] = "random string"

db = SQLAlchemy(app)

class users(db.Model):
    """create users table"""
   id = db.Column('id', db.Integer, primary_key = True)
   name = db.Column(db.String(100))
   email = db.Column(db.String(50), primary_key = True)
   pw = db.Column(db.String(200)) 
   
   def __init__(self, name, email, pw):
       self.name = name
       self.email = email
       self.pw = pw

class images(db.Model):
    """create images table"""
   #id = db.Column('id', db.Integer, primary_key = True)
   #users_id = db.Column(db.Integer, db.ForeignKey('users.id'))
   img_url = db.Column(db.String(100))
   #date = db.Column(db.DateTime) 

   #users = relationship("users")
   
   def __init__(self, img_url):
       #self.users_id = users_id
       self.img_url = img_url
       #self.date = date

@app.route('/')
def index():
    return '내 추억의 연장선 On the Line'
"""
#url 수정필요
@app.route('/', methods=['GET', 'POST'])
def home():
	""" Session control"""
	if not session.get('logged_in'):
		return render_template('index.html')
	else:
		if request.method == 'POST':
			useremail = getname(request.form['email'])
			return render_template('index.html', data=getfollowedby(useremail))
		return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
	"""Login Form"""
	if request.method == 'GET':
		return render_template('login.html')
	else:
		email = request.form['useremail']
		passw = request.form['password']
		try:
			data = User.query.filter_by(username=name, password=passw).first()
			if data is not None:
				session['logged_in'] = True
				return redirect(url_for('home'))
			else:
				return 'Dont Login'
		except:
			return "Dont Login"

@app.route('/register/', methods=['GET', 'POST'])
def register():
	"""Register Form"""
	if request.method == 'POST':
		new_user = User(username=request.form['username'], password=request.form['password'])
		db.session.add(new_user)
		db.session.commit()
		return render_template('login.html')
	return render_template('register.html')

@app.route("/logout")
def logout():
	"""Logout Form"""
	session['logged_in'] = False
	return redirect(url_for('home'))  """

@app.route('/img_trans', methods=['POST', 'GET'])
def img_trans():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            timedata = datetime()
            image_location = os.path.join(
                upload_folder + secure_filename(image_file.filename))
            image_file.save(image_location)
            upload_blob(image_location, timedata + 'image.png')

            back_removed = process.process(image_location)
            back_removed_name = 'back_removed_' + 'test.png'
            back_location = upload_folder + back_removed_name
            back_removed.save(back_location)
            upload_blob(back_location, timedata + 'back_removed.png')

            img = cv2.imread(back_location)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)

            total_color = 9

            img = color_quantization(img, total_color)

            blurred = cv2.bilateralFilter(
                img, d=7, sigmaColor=250, sigmaSpace=250)
            linist = cv2.bitwise_and(blurred, blurred, mask=edges)

            linist_name = upload_folder + 'linist_test.png'
            cv2.imwrite(linist_name, linist)
            linist = process.process(linist_name)
            linist.save(linist_name)
            upload_blob(linist_name, timedata + 'linist.png')

            # linist_url = 'https://storage.googleapis.com/linist_1/' + 파일명 <- url 이런 식으로 형성됨.
            new_image = images(linist_url)
		    db.session.add(new_image)
		    db.session.commit()
            removeAllFile(upload_folder)

            return send_file(linist, mimetype='image/jpeg')

        else:
            return jsonify({'Result': 'Fail'})

    else:
        return '내 추억의 연장선 On the Line'


@app.route('/background', methods=['POST', 'GET'])
def background():
    if request.method == 'POST':
        if request.files.get('image') and request.files.get('backImage'):
            timedata = datetime()
            image = request.files['image']
            backImage = request.files['backImage']

            image_name = secure_filename(image.filename)
            image_location = os.path.join(
                upload_folder + image_name)
            image.save(image_location)
            image = Image.open(image_location)
            image.convert("RGBA")
            imsize = image.size

            back_location = os.path.join(
                upload_folder + secure_filename(backImage.filename))
            backImage.save(back_location)
            backImage = Image.open(back_location)
            backImage = backImage.resize(imsize)

            datas = image.getdata()
            datas_back = backImage.getdata()

            finalData = list()

            for i in range(len(datas)):
                if datas[i][3] == 0:
                    finalData.append(datas_back[i])
                else:
                    finalData.append(datas[i])

            image.putdata(finalData)
            image.save(upload_folder + 'final.png')

            upload_blob(upload_folder + 'final.png', timedata + 'final.png')

            removeAllFile(upload_folder)

            return send_file(image, mimetype='image/jpeg')

        else:
            return jsonify({'Result': 'Fail'})
    else:
        return '내 추억의 연장선 On the Line'


if __name__ == '__main__':
>>>>>>> docker
    app.run(host='0.0.0.0', debug=True, port=int('333'))