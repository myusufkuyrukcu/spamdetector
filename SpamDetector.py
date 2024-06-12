from flask import Flask, request, render_template
from flask_restx import Api, Resource, fields, reqparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
api = Api(app, version='1.0', title='Spam Detector API', description='A simple spam detection API')

ns = api.namespace('spam', description='Spam operations')

model = None  # Global variable to hold the trained model
model_path = 'spam_detector_model.pkl'

# Swagger modelleri
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)

predict_model = api.model('PredictModel', {
    'message': fields.String(required=True, description='Message to classify as spam or ham')
})

# Web arayüzü için rota
@app.route('/')
def index():
    return render_template('index.html')

# Modeli eğitme uç noktası
@ns.route('/train')
@ns.expect(upload_parser)
class Train(Resource):
    @ns.expect(upload_parser)
    def post(self):
        """Train the spam detection model"""
        global model
        args = upload_parser.parse_args()
        csv_file = args['file']
        if not csv_file:
            return {'message': 'No file provided'}, 400

        # Geçici bir dosya olarak kaydet
        csv_path = os.path.join('uploads', csv_file.filename)
        csv_file.save(csv_path)
        
        # 1. CSV dosyasını oku
        try:
            data = pd.read_csv(csv_path, usecols=['class', 'message'], encoding='latin1')
        except UnicodeDecodeError as e:
            return {'message': f'Error decoding file: {e}'}, 400
        
        # 2. Veriyi hazırla
        X = data['message']
        y = data['class']
        
        # 3. Modeli eğit
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(X, y)
        
        # 4. Modeli kaydet
        model_full_path = os.path.abspath(model_path)
        print(f"Saving model to {model_full_path}")
        joblib.dump(model, model_full_path)
        
        return {'message': 'Model trained successfully'}, 200

# Metin tahmin etme uç noktası
@ns.route('/recognize')
@ns.expect(predict_model)
class Recognize(Resource):
    def post(self):
        """Predict if a message is spam or ham"""
        global model
        if not model:
            # Model yoksa önceden eğitilmiş modeli yükle
            if os.path.exists(model_path):
                model_full_path = os.path.abspath(model_path)
                print(f"Loading model from {model_full_path}")
                model = joblib.load(model_full_path)
            else:
                return {'message': 'Model not trained yet'}, 400
        
        message = api.payload['message']
        prediction = model.predict([message])[0]
        
        return {'prediction': prediction}, 200

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists(model_path):
        # Boş bir model dosyası oluştur
        initial_model = make_pipeline(CountVectorizer(), MultinomialNB())
        joblib.dump(initial_model, model_path)
    app.run(debug=True)
