# @author rmitsch
# @date 2017-08-06
#
import os
from flask import Flask
from flask import render_template
from flask import request, redirect, url_for, send_from_directory

app = Flask(__name__)
# Define version.
version = "0.1"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)