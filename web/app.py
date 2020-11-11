from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from pymongo import MongoClient
import bcrypt
#import requests
#import subprocess
import pandas as pd
import numpy as np
from scripts.facenet_tf2 import facenet_tf2
from scripts.verifyID import extract_face, verifyID, get_embedding
import cv2
import os, random
from PIL import Image


app = Flask(__name__)
api = Api(app)

client = MongoClient("mongodb://db:27017")
db = client.IDVerification
users = db["Users"]

# load model with weights
model = facenet_tf2()
model.load_weights('weights/nn4.small2.v1.h5')


#https://stackoverflow.com/questions/21732123/convert-true-false-value-read-from-file-to-boolean?lq=1
def str_to_bool(s):
    if s == 'true':
         return True
    elif s == 'false':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was

def get_photo(debug, username):

    frame = None

    if not debug:
        # get photo via webcam
        video_capture = cv2.VideoCapture(0)
        while True:
            _, frame = video_capture.read()
            cv2.imshow('Webcam', frame)

            # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
            # https://stackoverflow.com/questions/14494101/using-other-keys-for-the-waitkey-function-of-opencv/33555071#33555071
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

        # convert color space from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # get photo from folder
        filename = "images/" + username + "/" + random.choice(os.listdir("images/" + username + "/"))
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')

    return image

def InitizeDatabase():

    # password is only used for the purposes of testing
    password = u"test123"
    hashed_pw = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())

    # load saved embeddings
    db_init = pd.read_csv('data/db_init.csv')

    # add the password feature (dev purposes only)
    db_init["password"] = hashed_pw

    users.insert_many(db_init.to_dict(orient='records'))

def UserExist(username):
    if users.find({"username": username}).count()==0:
        return False
    else:
        return True

def verify_pw(username, password):
    if not UserExist(username):
        return False

    hashed_pw = users.find({
        "username": username
    })[0]["password"]

    if bcrypt.hashpw(password.encode('utf8'), hashed_pw)==hashed_pw:
        return True
    else:
        return False

def generateReturnDictionnary(status, msg):
    retJson = {
        "status" : status,
        "msg": msg
    }
    return retJson

def verifyCredentials(username, password):
    if not UserExist(username):
        return generateReturnDictionnary(301, "Invalid Username"), True

    correct_pw = verify_pw(username, password)
    if not correct_pw:
        return generateReturnDictionnary(302, "Invalid Password"), True

    return None, False


class Register(Resource):
    def post(self):
        postedData = request.get_json()

        username = postedData["username"]
        password = postedData["password"]
        debug = str_to_bool(postedData["debug"])

        if UserExist(username):
            retJson = {
                "status": 301,
                "msg": "Invalid username"
            }
            return jsonify(retJson)

        hashed_pw = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())

        image = get_photo(debug, username)
        filename = storeUserEmbedding(model, image, username)

        users.insert_one({
            "username": username,
            "password": hashed_pw,
            "embedding": filename
        })

        retJson = {
            "status": 200,
            "msg": "you are registered"
        }
        return jsonify(retJson)

class SignIn(Resource):
    def post(self):

        postedData = request.get_json()

        username = postedData["username"]
        password = postedData["password"]
        debug = str_to_bool(postedData["debug"])

        retJson, invalidPw = verifyCredentials(username, password)

        if invalidPw:
            return jsonify(retJson)

        image = get_photo(debug, username)
        verifyID(model, image, "data/" + username + ".npy")

        return jsonify(generateReturnDictionnary(200, "In the sign in"))


api.add_resource(Register, "/register")
api.add_resource(SignIn, "/sign_in")


@app.route('/')
def welcome():
    return 'Welcome to the API for the ID verification service. You first need to register with /register'

if __name__ == "__main__":
    InitizeDatabase()
    app.run(host='0.0.0.0')
