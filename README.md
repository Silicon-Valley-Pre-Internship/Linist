# 📸ON THE LINE : An extension of my memory

'On The Line📸' *is a mobile service that automates line drawing.*

 On the Line started with LINIST's vision to provide an opportunity to make memories as easy and comfortable as anyone wants.
 You can convert photos into line drawing images in any style you want, and share them with your friends by posting them on your account.
 
 
 ## ⚡TEAM _ LINIST⚡
 
    👩‍💻 KiHo Kim(Team Leader) : https://github.com/kiho0042
    
    🕵🏼‍♀️ Yujung Gil(Team Member) : https://github.com/fairyroad
    
    🙋 EunWoo Lee(Team Member) : https://github.com/rabbitew
    
    👩 Su-A Jang(Team Member) : https://github.com/su-aJ815
    
    🧙‍♂️ Yelim Jung(Team Member) : https://github.com/118dg




<나중에 여기에 결과 화면들 예쁘게 넣으면 될듯....>



## ✔Application Architecture✔
<p align="center"><img src="https://user-images.githubusercontent.com/74306759/124227821-3a76f400-db46-11eb-8aee-c70057833fdd.png"></p>

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


## ✔Tech Stack✔
### Version Control
    Git/Github

### Front End
    Hybrid App : React Native
    Web Server : Nginx

### Middleware
    WSGI : Gunicorn

### Back End
    Framework : Flask
    Image Processing : TensorFlow

### DB
    Client Data : MySQL
    Line Drawing Data : mongoDB

### Development Environment
    Visual studio code, colab
    docker
    Google Cloud Platform
    
|         Frontend         |      Backend      |         etc          |
| :----------------------: | :---------------: | :------------------: |
| <img alt="React Native" src="https://img.shields.io/badge/react_native-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB"/> <img alt="NodeJS" src="https://img.shields.io/badge/node.js-%2343853D.svg?style=for-the-badge&logo=node-dot-js&logoColor=white"/> | ![Flask](https://img.shields.io/badge/flask-v1.1.2-green?logo=flask) ![Gunicorn](https://img.shields.io/badge/gunicorn-v20.0.4-darkgreen?logo=gunicorn) <img src="https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=MongoDB&logoColor=white"/> ![MySQL](https://img.shields.io/badge/mysql-v4.2.11-blue?logo=mysql) | ![Docker](https://img.shields.io/badge/docker-v20.10.2-blue?logo=docker) ![Nginx](https://img.shields.io/badge/Nginx-v1.14.0-brightgreen?logo=nginx) ![github](https://img.shields.io/badge/github-gray?logo=github) ![VScode](https://img.shields.io/badge/VScode-v1.52.1-blue?logo=visual-studio-code) ![Google Cloud Platform](https://img.shields.io/badge/Google_Cloud_Platform-VM_instance-red?logo=gcp) [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naereen/badges)|
