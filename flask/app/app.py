from flask import Flask, request, send_file, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io as IO
import os
from skimage import io
import cv2

upload_folder = 'static'

back_model = tf.keras.models.load_model('modelcombined_04_0.238711.h5')

def background_removal(imgpath, img):
    if imgpath:
        im = io.imread(imgpath)
    else:
        im = img.copy()

    im = cv2.resize(im[:,:,0:3], (256,256))
    img = np.array(im)/255
    img = img.reshape((1,)+img.shape)
    pred = back_model.predict(img)

    p = pred.copy()
    p = p.reshape(p.shape[1:-1])

    p[np.where(p>.25)] = 1
    p[np.where(p<.25)] = 0

    im[:,:,0] = im[:,:,0]*p 
    im[:,:,0][np.where(p!=1)] = 255
    im[:,:,1] = im[:,:,1]*p 
    im[:,:,1][np.where(p!=1)] = 255
    im[:,:,2] = im[:,:,2]*p
    im[:,:,2][np.where(p!=1)] = 255

    return im

def load_img(image_location):
    img = io.imread(image_location)
    img = cv2.resize(img[:,:,0:3], (256,256), interpolation=cv2.INTER_AREA)
    return img

def save_img(img,image_name):
    image_location = os.path.join(upload_folder, image_name)
    cv2.imwrite(image_location, img)
    return image_location

def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

app = Flask(__name__)

@app.route('/')
def index():
    return '내 추억의 연장선 On the Line'

@app.route('/img_trans', methods = ['POST', 'GET'])
def img_trans():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(upload_folder, image_file.filename)
            image_file.save(image_location)

            new_image = load_img(image_location)
            new_image_name = 'new_'+image_file.filename+'.jpg'
            new_image_location = save_img(new_image, new_image_name)

            back_removed = background_removal(imgpath=new_image_location, img=None)
            back_removed_name = 'back_removed_'+image_file.filename
            back_location = save_img(back_removed, back_removed_name)

            img = cv2.imread(back_location)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray,5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)

            total_color = 9

            img = color_quantization(img, total_color)

            blurred = cv2.bilateralFilter(img, d=7, sigmaColor=250, sigmaSpace=250)
            linist = cv2.bitwise_and(blurred, blurred, mask =edges)

            linist_name = 'linist_' + image_file.filename +'.jpg'
            linist_location = cv2.save_img(linist, linist_name)

            return send_file(linist, mimetype='image/jpeg')

        else:
            return jsonify({'Result' : 'Fail'})

    else:
        return '내 추억의 연장선 On the Line'

@app.route('/background', methods = ['POST', 'GET'])
def background():
    if request.method == 'POST':
        if request.files.get('Image') and request.files.get('backImage'):
            image = request.files['Image'].read()
            image = Image.open(IO.ByteIO(image)).convert('RGB')

            backImage = request.files['backImage'].read()
            backImage = Image.open(IO.ByteIO(backImage)).convert('RGB')

            image_location = os.path.join(upload_folder, image.filename)
            image.save(image_location)

            back_location = os.path.join(upload_folder, backImage.filename)
            backImage.save(back_location)

            new_image = load_img(image_location)
            back_image = load_img(back_location)

            mark = np.copy(image)

            blue_threshold = 200
            green_threshold = 200
            red_threshold = 200
            bgr_threshold = [blue_threshold, green_threshold, red_threshold]

            thresholds = (image[:,:,0] < bgr_threshold[0]) | (image[:,:,1] < bgr_threshold[1]) | (image[:,:,2] < bgr_threshold[2])
            mark[thresholds] = [0,0,0]

            masked = cv2.bitwise_and(img2, mark)
            masked2 = cv2.bitwise_and(image, 255-mark)

            final = masked+masked2

            final_name = 'back_' + image.filename +'.jpg'
            linist_location = cv2.save_img(final, final_name)

            return send_file(final, mimetype='image/jpeg')

        else:
            return jsonify({'Result' : 'Fail'})
    else:
        return '내 추억의 연장선 On the Line'

if __name__ == '__main__':
    app.run(host ='0.0.0.0', debug=True, port ='333')