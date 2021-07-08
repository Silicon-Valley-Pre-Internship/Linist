import tensorflow_hub as hub
from flask import Flask, request, send_file, jsonify
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import PIL
import numpy as np
import io
import json

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

app = Flask(__name__)

@app.route('/')
#def index():
#    return '내 추억의 연장선 On the Line'

#@app.route('/image_trans', methods = ['POST', 'GET'])
#def send_image():
#    if request.method == 'POST':
#        if request.files.get('Image'):

def index():
    if 0==0:
        if 0==0:
            #image = request.files['myImage'].read()
            image = tf.keras.utils.get_file('image_.jpg','https://mblogthumb-phinf.pstatic.net/MjAyMDA0MTdfMTE0/MDAxNTg3MTEyNjg1MTgy.MODHxVr9PZlAKeYQ8M5M3wqU2RHazeaAdyF9wtSbMp4g.msjj2BW_2OPRRWv57WOO5thpL139Dx5PJnwt_bNuKRYg.PNG.thirdsky30/CropArea0008.png?type=w800')
            image = load_img(image)
            #image = tf.image.sobel_edges(image)
            #image = image[...,1]/4+0.5

            style = tf.keras.utils.get_file('style_.jpg','https://image.idus.com/image/files/d2034014c2b448f1916a2d70f3561783_720.jpg')
            style = load_img(style)

            numpy_image = hub_module(tf.constant(image), tf.constant(style))[0]
            numpy_image = tensor_to_image(numpy_image)
            
            byte_io = io.BytesIO()
            numpy_image.save(byte_io,'jpeg')
            byte_io.seek(0)
            numpy_image.show()

            return send_file(numpy_image, mimetype='image/jpeg')

        else:
            return jsonify({'Result' : 'Fail'})

    else:
        return '내 추억의 연장선 On the Line'


@app.route('/save_image', methods=['POST', 'GET'])
def save_image():
    if request.method == "POST":
        success = False
        image_data = request.get_json()
        author = image_data['author']
        name = image_data['name']
        src = image_data['url']
        date = datetime()

        sql = f"INSERT INTO trans_images (author, name, src, date) VALUES('{author}', '{name}', '{src}', '{date}');"
        success = conn_db(sql, "insert")
        return jsonify({"success": success})
    else:
        return "hello world"

@app.route('/get_album', methods=['POST', 'GET'])
def get_album():
    if request.method == "GET": 
        sql = "SELECT id, author, name, src FROM trans_images;"
        data = conn_db(sql, "select")
        return json.dumps(data)
    else:
        return "hello world"


def conn_db(sql, sql_type):
    import pymysql

    conn = pymysql.connect(host='', user='', password='', db='') # addr, user, password, db
    curs = conn.cursor(pymysql.cursors.DictCursor)
    curs.execute(sql)
    if sql_type == "insert":
        conn.commit()
        conn.close()
        return True
    elif sql_type == "select":
        rows = curs.fetchall()
        conn.close()
        return rows

    return False

def datetime():
    import datetime
    now = datetime.datetime.now()
    return now.strftime('%Y%m%d%H%M%S')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)