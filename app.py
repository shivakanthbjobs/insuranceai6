from flask import Flask, redirect, url_for, session
from flask_oauth import OAuth
from flask import Flask, render_template,request
from flask import flash
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import os
import cv2
import base64
import dlib
import numpy as np
import argparse
import inception_resnet_v1
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from flask import request
from flask import jsonify
from flask import Flask
from imageio import imread
import base64
import io
import keras
import sys
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from PIL import Image

 
import configparser 

 

GOOGLE_CLIENT_ID = ''
GOOGLE_CLIENT_SECRET = ''
REDIRECT_URI = '/oauth2callback'  # one of the Redirect URIs from Google APIs console
 
SECRET_KEY = 'development key'
DEBUG = True
 
app = Flask(__name__)
app.debug = DEBUG
app.secret_key = SECRET_KEY
oauth = OAuth()
 
google = oauth.remote_app('google',
                          base_url='https://www.google.com/accounts/',
                          authorize_url='https://accounts.google.com/o/oauth2/auth',
                          request_token_url=None,
                          request_token_params={'scope': 'https://www.googleapis.com/auth/userinfo.email',
                                                'response_type': 'code'},
                          access_token_url='https://accounts.google.com/o/oauth2/token',
                          access_token_method='POST',
                          access_token_params={'grant_type': 'authorization_code'},
                          consumer_key=GOOGLE_CLIENT_ID,
                          consumer_secret=GOOGLE_CLIENT_SECRET)
 
 
 

@app.route('/oauthgmail')
def index():
    access_token = session.get('access_token')
    if access_token is None:
        return redirect(url_for('login'))
 
    access_token = access_token[0]
    from urllib.request import Request, urlopen, URLError
 
    headers = {'Authorization': 'OAuth '+access_token}
    req = Request('https://www.googleapis.com/plus/v1/people/me',
                  None, headers)
    try:
        res = urlopen(req)
    except e:
        if e.code == 401:
            # Unauthorized - bad token
            session.pop('access_token', None)
            return redirect(url_for('login'))
        return res.read()
 
    #return res.read()
    return render_template('index.html')
 
@app.route('/')
def home():
    return render_template('UserDetails.html')
 

 
@app.route('/getQuote', methods=['GET', 'POST'])
def getPolicyQuote():
    form = request.form
    if request.method == 'POST':
       Name = request.form['Name']
       Email = request.form['Email']
       Age = request.form['Age']
       Height = request.form['Height']
       Gender = request.form['Gender']
       Smoker = request.form['Smoker']
       Drinker = request.form['Drinker']
       Health = request.form['Fitness']
       Weight = request.form['Weight']
       print('Sudhir Test Form fields')
	   
       Premium = PremiumCalculation(200000, Age, Height, Gender, Smoker, Drinker, Health, Weight)
       PremiumGold = PremiumCalculation(400000, Age, Height, Gender, Smoker, Drinker, Health, Weight)
       PremiumPlatinum = PremiumCalculation(1000000, Age, Height, Gender, Smoker, Drinker, Health, Weight)
       print('Premium is - '+ str(Premium))
       print('PremiumGold is - '+ str(PremiumGold))
       print('PremiumPlatinum is - '+ str(PremiumPlatinum))
	   
	   
	   
	   

	
    print('Sudhir form ends')
    return render_template('QuoteDetails.html', **locals())
 

def PremiumCalculation(PV, Age, Height, Gender, Smoker, Drinker, Health, Weight):
       ageRange = 25
	   
       if int(Age) >= 25:
          factor = int((int(Age)-25)/10)
          ageRange = 25 + (factor+1)*10
       else:
          ageRange = 25
		 
	   
       print(ageRange)
       #premiumKey=str(ageRange)+bmicategory+Smoker+Drinker+Gender+Health
       #print(premiumKey)
	   
       #config = configparser.RawConfigParser()
       #config.read('ConfigFile.properties')
       #Premium = config.get('PolicyPremiumSection', premiumKey)
       #print(Premium)
       bmicategory = (float(Weight)) / ((float(Height)/100)* (float(Height)/100))
	   
       bmicategoryValue = 0
       print('bmicategory '+ str(bmicategory))
       if bmicategory < 18:
          bmicategoryValue = 5
       elif bmicategory > 27:
          bmicategoryValue = 10
       else:
          bmicategoryValue = 0
		  
       print(str(bmicategoryValue))
	   
       #Constants	 
       #PremiumValue
       #PV = 400000
       #Policy Claim Rate
       PCR = 50
       #Policy Per Unit Value
       PPUV = 100
       #Policy Factor
       PF = 500
       #Tax Rate
       TR = 18
	   
       #Premium Factor
       PmF = ((PV/PPUV)+((PV/PPUV)*PCR/100))
	   
       print('PmF - '+ str(PmF))
	   
       #Risk Factor for SMoker
       RFSmoker = 0
       if Smoker == 'Y':
          RFSmoker = 20
       print('Smoker'+str(RFSmoker))
	   
       #Risk Factor if Drinker
       RFDrinker = 0
       if Drinker == 'Y':
          RFDrinker = 15

       print('Drinker'+str(RFDrinker))

       #Risk Factor on Gender
       RFGender = 0	   
       if Gender == 'F':
         RFGender = 5
       print('Gender ' + str(RFGender))
	   
       #Risk Factor for Health
       RFHealth = 0
       if Health == 'MH':
           RFHealth = 5
       elif Health == 'UnH':
           RFHealth = 10
	   
	   
       print('Health ' + str(RFHealth))
   	   
	   
       #Risk Factor
       print('Age risk factor is ' + str((int(ageRange/10))*5))
       RF = ((((int(ageRange/10))*5)) + bmicategoryValue + RFSmoker + RFDrinker + RFGender + RFHealth)
       print('Risk Factor calulated is ' + str(RF))
	   
       #TAX
       Tax = PmF * TR / 100
       print('Tax is ' + str(Tax))
	   
       #Premium
       Premium = (PmF + (PmF * RF/100) + PF + Tax )
	   
       print('Premium is - '+ str(Premium))
	   
       return Premium
 
@app.route('/login')
def login():
    callback=url_for('authorized', _external=True)
    return google.authorize(callback=callback)
 
 
 
@app.route(REDIRECT_URI)
@google.authorized_handler
def authorized(resp):
    access_token = resp['access_token']
    session['access_token'] = access_token, ''
    return redirect(url_for('index'))
 
@google.tokengetter
def get_access_token():
    return session.get('access_token') 
 

@app.route('/predict',methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    pos = encoded.index(',')
    encoded = encoded[pos+1:]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    image = np.array(image)
    image2= image.copy()
    print("image2",image2)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "--M", default="./ageGendermodels", type=str, help="Model Path")
    args = parser.parse_args()
    sess, age, gender, train_mode,images_pl = load_network(args.model_path)
    ageGender= getAgeGender(sess,age,gender,train_mode,images_pl,image).split(',')

    rate = rate_factor(image2)
    print("rate",rate)

    response = jsonify({  
            'age' : ageGender[0],
            'gender' : ageGender[1],
            'rate' : rate})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    


def face_remap(shape):
    remapped_image = cv2.convexHull(shape)
    return remapped_image

def convex_hull(shape,image):
    shape = np.array(shape,dtype="int")
    out_face = np.zeros_like(image)
    remapped_shape = np.zeros_like(shape) 
    feature_mask = np.zeros((image.shape[0], image.shape[1]))   
    remapped_shape = face_remap(shape)
    cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
    feature_mask = feature_mask.astype(np.bool)
    out_face[feature_mask] = image[feature_mask]
    return out_face
    
def crop_pts(xy, frame):
    pts = np.array(xy, dtype=np.int32)
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = frame[y:y+h, x:x+w]
    return croped

def wrinkle_density(img):
    pxl_cnt = 0
    wht_cnt = 0
    for od in img:
        for pxl in od:
            if pxl == 255:
                wht_cnt = wht_cnt+1
            pxl_cnt = pxl_cnt+1
    return wht_cnt/pxl_cnt

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rate_factor(frame):
    print("frame" , frame)
    predictor_path = r"shape_predictor_68_face_landmarks.dat"
    crowfeet1 = [1,18,2,37]
    crowfeet2 = [27,17,46,16]
    eye_bag1  = [1,42,41,32,3]
    eye_bag2  = [17,47,48,36,15]
    teeth = [61,62,63,64,65,66,67,68]
    mouth = [49,50,51,52,53,54,55,56,57,58,59,60]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    cpy = frame.copy()
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(frame_resized, 1)
    print("dets",dets)
    if len(dets) > 0:
        for k, d in enumerate(dets):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)
            i = 1
            cf1 = []
            c1 = 0
            cf2 = []
            c2 = 0
            eb1 = []
            e1 = 0
            eb2 = []
            e2 = 0
            mth = []
            m = 0   
            tth = []
            t = 0
            shape2 = []
            s = 0
            for (x, y) in shape:
                shape2.insert(s,[int(x/ratio),int(y/ratio)])    
                if i in teeth:
                    tth.insert(t,[int(x/ratio),int(y/ratio)])
                    t = t+1
                if i in mouth:
                    mth.insert(c2,[int(x/ratio),int(y/ratio)])
                    m = m+1
                if i in crowfeet2:
                    cf2.insert(c2,[int(x/ratio),int(y/ratio)])
                    c2 = c2+1
                if i in crowfeet1:
                    cf1.insert(c1,[int(x/ratio),int(y/ratio)])
                    c1 = c1+1
                if i == 19:
                    x1 = int(x/ratio)
                    y1 = int(y/ratio)-5
                if i == 26:
                    x2 = int(x/ratio)
                    y2 = int(y/ratio) - 53
                if i in eye_bag1:
                    eb1.insert(e1,[int(x/ratio),int(y/ratio)])
                    e1 = e1+1
                if i in eye_bag2:
                    eb2.insert(e2,[int(x/ratio),int(y/ratio)])
                    e2 = e2+1
                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 1 , (255, 125, 110), -1)
                i = i+1
                s = s+1
            fh  = frame[y2:y1, x1:x2]
            cf1 = crop_pts(cf1,cpy)
            cf2 = crop_pts(cf2,cpy)
            eb1 = crop_pts(eb1,cpy)
            eb2 = crop_pts(eb2,cpy)
            mth = convex_hull(mth,cpy)
            tth = convex_hull(tth,cpy)
            
    fcn   = cv2.Canny(fh,  120, 50, L2gradient=False)
    cf1cn = cv2.Canny(cf1, 120, 50, L2gradient=False)
    cf2cn = cv2.Canny(cf2, 120, 50, L2gradient=False)
    eb1cn = cv2.Canny(eb1, 120, 50, L2gradient=False)
    eb2cn = cv2.Canny(eb2, 120, 50, L2gradient=False)
    cv2.imshow("fh",fcn)
    cv2.imshow("cf1",cf1)
    cv2.imshow("cf2",cf2)
    return(1+(wrinkle_density(fcn)+wrinkle_density(cf1cn)+wrinkle_density(cf2cn)+wrinkle_density(eb1cn)+wrinkle_density(eb2cn)))

    
#AGE GENDER


def get_args():

    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "

                                                 "and estimates age and gender for the detected faces.",

                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--weight_file", type=str, default=None,

                        help="path to weight file (e.g. weights.18-4.06.hdf5)")

    parser.add_argument("--depth", type=int, default=16,

                        help="depth of network")

    parser.add_argument("--width", type=int, default=8,

                        help="width of network")

    args = parser.parse_args()

    return args

 

 

def getAgeGender(sess,age,gender,train_mode,images_pl,img):

    args = get_args()

    depth = args.depth

    k = args.width

 

    # for face detection

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    fa = FaceAligner(predictor, desiredFaceWidth=160)

 

    # load model and weights

    img_size = 160

 

    # capture video

    # cap = cv2.VideoCapture(0)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

   

 

    # get video frame

    # ret, img = cap.read()

 

    # if not ret:

    #     print("error: failed to capture image")

    #     return -1

 

    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_h, img_w, _ = np.shape(input_img)

 

    # detect faces using dlib detector

    detected = detector(input_img, 1)

    faces = np.empty((len(detected), img_size, img_size, 3))

 

    for i, d in enumerate(detected):

        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()

        xw1 = max(int(x1 - 0.4 * w), 0)

        yw1 = max(int(y1 - 0.4 * h), 0)

        xw2 = min(int(x2 + 0.4 * w), img_w - 1)

        yw2 = min(int(y2 + 0.4 * h), img_h - 1)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)

        faces[i, :, :, :] = fa.align(input_img, gray, detected[i])

        # faces[i,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

    #

    if len(detected) > 0:

        # predict ages and genders of the detected faces

        ages,genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})

   

    # draw results

    for i, d in enumerate(detected):

        label = "{}, {}".format(int(ages[i]), "F" if genders[i] == 0 else "M")
        
        return label
        #draw_label(img, (d.left(), d.top()), label)

 

   # cv2.imshow("result", img)

 

def load_network(model_path):

    sess = tf.Session()

    images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')

    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)

    train_mode = tf.placeholder(tf.bool)

    age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,

                                                                 phase_train=train_mode,

                                                                 weight_decay=1e-5)

    gender = tf.argmax(tf.nn.softmax(gender_logits), 1)

    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)

    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)

    init_op = tf.group(tf.global_variables_initializer(),

                       tf.local_variables_initializer())

    sess.run(init_op)

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(model_path)

    if ckpt and ckpt.model_checkpoint_path:

        saver.restore(sess, ckpt.model_checkpoint_path)

        print("restore model!")

    else:

        pass

    return sess,age,gender,train_mode,images_pl
 
def main():
    app.run()
 
 
if __name__ == '__main__':
    main()
