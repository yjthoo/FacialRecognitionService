from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from pymongo import MongoClient
import bcrypt
import requests
import subprocess
import jsonify

app = Flask(__name__)
api = Api(app)

client = MongoClient("mongodb://db:27017")
db = client.ImageRecognition
users = db["Users"]


def UserExist(username):
    if users.find({"Username": username}).count()==0:
        return False
    else:
        return True

def verify_pw(username, password):
    if not UserExist(username):
        return False

    hashed_pw = users.find({
        "Username": username
    })[0]["Password"]

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

        if UserExist(username):
            retJson = {
                "status": 301,
                "msg": "Invalid username"
            }
            return jsonify(retJson)

        hashed_pw = bcrypt.hashed_pw(password.encode(utf8), bcrypt.gensalt())

        users.insert_one({
            "Username": username,
            "Password": hashed_pw,
            "Tokens": 4
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

        retJson, invalidPw = verifyCredentials(username, password)

        if invalidPw:
            return jsonify(retJson)

        # TODO: Facial recognition

        return jsonify(generateReturnDictionnary(200, "In the sign in"))


api.add_resource(Register, "/register")
api.add_resource(SignIn, "/sign_in")

if name == "__main__":
    app.run(host='0.0.0.0')
