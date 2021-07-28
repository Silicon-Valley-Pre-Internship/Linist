from flask import Flask, request, send_file, jsonify
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

            return linist_url

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

            return final_url

        else:
            return jsonify({'Result': 'Fail'})
    else:
        return '내 추억의 연장선 On the Line'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=int('333'))