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

#@app.route('/send_image', methods = ['POST', 'GET'])
#def send_image():
#    if request.method == 'POST':
#        if request.files.get('Image'):

def index():
    if 0==0:
        if 0==0:
            #image = request.files['myImage'].read()
            image = tf.keras.utils.get_file('image.jpg','https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Lee_Jong-suk_March_2018.png/250px-Lee_Jong-suk_March_2018.png')
            image = load_img(image)

            style = tf.keras.utils.get_file('style.jpg','https://image.idus.com/image/files/794d59d229704961ab8f17d0ce36e95c_512.jpg')
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


if __name__ == '__main__':
    app.run(host ='0.0.0.0', debug=True, port ='333')