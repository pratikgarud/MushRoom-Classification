from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('Mushroom_Model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['GET','POST'])
def classify():
    if request.method == 'POST':
        CapSurface = request.form.get('capsurface')
        GillSize = request.form.get('gillsize')
        VeilColor = request.form.get('veilcolor')
        RingNo = request.form.get('ringnum')
        RingType = request.form.get('ringtype')
        StalkType = request.form.get('stalktype')
        all_feat = np.array([CapSurface,GillSize,VeilColor,RingNo,RingType,StalkType])
        pre = model.predict([all_feat])
        pred = np.asscalar(pre)
    return render_template('index.html', output=pred)

if __name__=='__main__':
    app.run(debug=True)