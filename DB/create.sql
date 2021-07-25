DROP DATABASE IF EXISTS linist_db;
CREATE DATABASE linist_db DEFAULT CHARSET=utf8 COLLATE=UTF8_BIN;

USE linist_db;
DROP TABLE IF EXISTS users;
CREATE TABLE users(
    id int(10) NOT NULL AUTO_INCREMENT,
    name varchar(25) NOT NULL,
    email varchar(40) NOT NULL,
    pw varchar(20) NOT NULL,
    PRIMARY KEY(id, email)
);

DROP TABLE IF EXISTS image;
CREATE TABLE image(
    id int(10) NOT NULL AUTO_INCREMENT,
    users_id int(10) NOT NULL,
    bucket_link varchar(50) NOT NULL,
    date date NOT NULL,
    PRIMARY KEY(id),
    FOREIGN KEY(users_id) REFERENCES users(id) on update cascade on delete cascade
);

