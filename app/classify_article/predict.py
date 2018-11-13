#!/usr/bin/env python3

import connexion

from sklearn.externals import joblib

classifier = joblib.load('../../output/model.pkl')


def post_predictions(query):
    predictions = []
    for item in query:
        text = item['text']
        category = classifier.predict([text])[0]
        predictions.append({"category": category, "text": text})
    return predictions


predict = connexion.App(__name__)
predict.add_api('swagger.yaml')

if __name__ == '__main__':
    predict.run(port=8001, server='gevent')
