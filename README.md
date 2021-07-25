#ON THE LINE : An extension of my memory

'On The Line📸' *is a mobile automation service changes image to line drawing. 

If you give photos to the service, you can print new photos turned into line drawing. Also you can keep the converted images in application and share it with friends allowed to approach. 

Here's an example of the service.
 
 
 ## ⚡TEAM _ LINIST⚡
  
  *On the Line started with LINIST's vision to provide an opportunity to make memories as easy and comfortable as anyone wants.*
  
 
    👩‍💻 KiHo Kim(Team Leader) : https://github.com/kiho0042
    
    🕵🏼‍♀️ Yujung Gil(Team Member) : https://github.com/fairyroad
    
    🙋 EunWoo Lee(Team Member) : https://github.com/clairew99
    
    👩 Su-A Jang(Team Member) : https://github.com/su-aJ815
    
    🧙‍♂️ Yelim Jung(Team Member) : https://github.com/118dg



## ✔Application Feature✔

### Main Feature (PoC)
    1. Line drawing by removing the background of the picture.
    2. Color by part (hair or clothing).
    3. Change the background of the results to the desired background

### Additional Feature
    1. Create posts by account and manage them by date.
       cf. Instagram Feeds
    2. only makes it(posts) visible to people I want to share using the Kakao Talk API
       cf. Everytime timetable sharing function

## ✔UI/UX PROTOTYPE✔
    XD prototype link : https://xd.adobe.com/view/135fad62-dd04-42ee-849d-7bde6faab676-460c/?fullscreen)
    
 
## ✔Service Flow✔

    1.When a user launches a mobile app, it shows a login or membership page depending on whether he or she is registered or not.
    
    2.After logging in, the page where you can upload photos first appears.
    
    3.This is to encourage users who first access the service to follow the process of obtaining line drawing images naturally.
    
    4.When a user uploads a picture, a black-and-white line image, which is named "Nukki," is created first based on image processing using tensorflow.
    
    5.Here, the user can select and apply the desired background image and can also color it.
    
    6.Completed copies that have been customized according to the user's intentions can be shared on SNS or stored in personal albums.
    
    7.Personal albums can be shared among allowed friends, and for this purpose, friend add-ons will be implemented.
    
    8.The album is expressed in the form of an "instagram feed" where you can scroll down to see posts and photos.


## ✔Tech Stack✔
### Version Control
    Git/Github

### Frontend
    Hybrid App : React Native
    Web Server : Nginx

### Gateway
    WSGI : Gunicorn

### Backend
    Framework : Flask
    Image Processing : TensorFlow

### DB
    Client and line drawing Data : MySQL

### Development Environment
    Visual studio code, colab
    Docker
    Google Cloud Platform
    
|         Frontend         |      Backend      |         etc          |
| :----------------------: | :---------------: | :------------------: |
| <img alt="React Native" src="https://img.shields.io/badge/react_native-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB"/> <img alt="NodeJS" src="https://img.shields.io/badge/node.js-%2343853D.svg?style=for-the-badge&logo=node-dot-js&logoColor=white"/> | ![Flask](https://img.shields.io/badge/flask-v1.1.2-green?logo=flask) ![Gunicorn](https://img.shields.io/badge/gunicorn-v20.0.4-darkgreen?logo=gunicorn) <img src="https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=MongoDB&logoColor=white"/> ![MySQL](https://img.shields.io/badge/mysql-v4.2.11-blue?logo=mysql) | ![Docker](https://img.shields.io/badge/docker-v20.10.2-blue?logo=docker) ![Nginx](https://img.shields.io/badge/Nginx-v1.14.0-brightgreen?logo=nginx) ![github](https://img.shields.io/badge/github-gray?logo=github) ![VScode](https://img.shields.io/badge/VScode-v1.52.1-blue?logo=visual-studio-code) ![Google Cloud Platform](https://img.shields.io/badge/Google_Cloud_Platform-VM_instance-red?logo=gcp) [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naereen/badges)|


## ✔Application Architecture✔
<p align="center"><img src="https://user-images.githubusercontent.com/74306759/126728093-b7f1dbb1-3a29-487f-827a-e3d51126150d.png"></p>


