from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from pymongo import MongoClient
import bcrypt
import requests
import subprocess

app = Flask(__name__)
api = Api(app)

client = MongoClient("mongodb://db:27017") #, username='user', password='password')
db = client.IDVerification
users = db["Users"]

def InitizeDatabase():

    password = u"test"
    hashed_pw = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())

    users_list = [
        {
            "username": 'mindy_kaling',
            "password": hashed_pw
        },
        {
            'username': 'madonna',
            "password": hashed_pw
        },
        {
            'username': 'elton_john',
            "password": hashed_pw
        },
        {
            'username': 'ben_afflek',
            "password": hashed_pw
        },
        {
            'username': 'jerry_seinfeld',
            "password": hashed_pw
        }
    ]

    users.insert_many(users_list)

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

        if UserExist(username):
            retJson = {
                "status": 301,
                "msg": "Invalid username"
            }
            return jsonify(retJson)

        hashed_pw = bcrypt.hashpw(password.encode('utf8'), bcrypt.gensalt())

        users.insert_one({
            "username": username,
            "password": hashed_pw,
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


@app.route('/')
def welcome():
    return 'Welcome to the API for the ID verification service. You first need to register with /register'

if __name__ == "__main__":
    InitizeDatabase()
    app.run(host='0.0.0.0')
