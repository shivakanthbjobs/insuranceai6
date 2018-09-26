from flask import Flask, redirect, url_for, session
from flask_oauth import OAuth
from flask import Flask, render_template,request
from flask import flash
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import configparser 

 

GOOGLE_CLIENT_ID = '346115038766-gueouf98f2kjdn9qudipgolko0kmscfu.apps.googleusercontent.com'
GOOGLE_CLIENT_SECRET = 'SX9Y6XNhDecgC3chMQVofjGG'
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
       Health = request.form['Health']
       print('Sudhir Test Form fields')
	   
       bmicategory='B0'
       factor = int((int(Age)-25)/10)
       ageRange = 25 + (factor+1)*10
       print(ageRange)
       premiumKey=str(ageRange)+bmicategory+Smoker+Drinker+Gender+Health
       print(premiumKey)
       config = configparser.RawConfigParser()
       config.read('ConfigFile.properties')
       Premium = config.get('PolicyPremiumSection', premiumKey)
       print(Premium)

	
    print('Sudhir form ends')
    return render_template('QuoteDetails.html', **locals())
 
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
 

 
def main():
    app.run()
 
 
if __name__ == '__main__':
    main()
