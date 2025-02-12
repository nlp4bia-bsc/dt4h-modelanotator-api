from flask import Flask, request, jsonify
from flasgger import Swagger
from app.models.dictionary_baseline import DictionaryLookupModel

import os
import logging

app = Flask(__name__)
swagger = Swagger(app)


model_language = str(os.getenv("LANGUAGE", "EN")).lower()

language = os.getenv("DICTIONARY_FILE", "english")

# Initialize the model
model = DictionaryLookupModel("./" + str(language) + "_entities.csv")

@app.route('/process_text', methods=['POST'])
def process_text():
    """
    Process text using NER Dictionary Lookup
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: text_input
          required:
            - text
          properties:
            text:
              type: string
              description: The text to process
    responses:
      200:
        description: Processed text with NER annotations
    """
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Input must be a dictionary with 'content' key"}), 400

    if not "content" in data.keys():
        return jsonify({"error": "Input must be a dictionary with 'content' key"}), 400

    data = data["content"]

    text = data.get('text')

    id = data.get('id')

    if not text:
        return jsonify({"error": "'text' is required"}), 400

    result = model.predict(text=text, app=app, id=id, language=model_language)
    return jsonify(result)

@app.route('/process_bulk', methods=['POST'])
def process_bulk():
    """
    Process multiple texts using NER Dictionary Lookup
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: bulk_input
          type: array
          items:
            type: object
            required:
              - text
            properties:
              text:
                type: string
                description: The text to process
    responses:
      200:
        description: Processed texts with NER annotations
    """
    data = request.json

    if not isinstance(data, dict):
        return jsonify({"error": "Input must be a dictionary with 'content' key"}), 400

    if not "content" in data.keys():
        return jsonify({"error": "Input must be a dictionary with 'content' key"}), 400

    data = data["content"]

    if not isinstance(data, list):
        return jsonify({"error": "Input must be a list of objects"}), 400

    results = []
    for item in data:
        text = item.get('text')

        id = item.get('id')

        if not text:
            return jsonify({"error": "Each item must contain 'text'"}), 400

        result = model.predict(text, app, id=id, language=model_language)
        results.append(result)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
