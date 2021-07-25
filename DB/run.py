from flask import Flask, request, flash, url_for, redirect, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:ewha1886@localhost/linist_db'
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

@app.route('/')
def show_all():
   return render_template('show_all.html', users = users.query.all() )

@app.route('/new', methods = ['GET', 'POST'])
def new():
   if request.method == 'POST':
      if not request.form['name'] or not request.form['email'] or not request.form['pw']:
         flash('Please enter all the fields', 'error')
      else:
         user = users(request.form['name'], request.form['email'], 
            request.form['pw'])
         
         db.session.add(user)
         db.session.commit()
         flash('Record was successfully added')
         return redirect(url_for('show_all'))
   return render_template('new.html')

if __name__ == '__main__':
   db.create_all()
   app.run(debug = True)