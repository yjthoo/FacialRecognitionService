# function for face detection with mtcnn
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims

# extract a single face from a given photograph
def extract_face(filename, required_size=(96, 96)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
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

# check if the embeddings have a high similarity or not
def verifyID(ref_img, cap_img, threshold = 0.5):

    if np.linalg.norm(ref_img - cap_img) < threshold:
        return True

    return False
