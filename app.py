from flask import Flask, render_template, request
from flask import redirect
from flask import url_for

app = Flask(__name__)



if __name__== '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)