# function for face detection with mtcnn
import numpy as np
from numpy import expand_dims
import cv2
from PIL import Image
from mtcnn.mtcnn import MTCNN

# extract a single face from a given photograph
def extract_face(image, debug = False, required_size=(96, 96)):

    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)

    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

# obtain and save the embeddings of a new user
def storeUserEmbedding(model, frame, username):

    cap_face = extract_face(frame)
    cap_emb = get_embedding(model, cap_face)

    filename = "data/" + username + ".npy"
    np.save(filename, cap_emb)
    return filename

# check if the embeddings have a high similarity or not
def verifyID(model, frame, ref_path, threshold = 0.5):

    cap_face = extract_face(frame)
    cap_emb = get_embedding(model, cap_face)
    ref_emb = np.load(ref_path, allow_pickle = True)

    if np.linalg.norm(ref_emb - cap_emb) < threshold:
        return True

    return False
