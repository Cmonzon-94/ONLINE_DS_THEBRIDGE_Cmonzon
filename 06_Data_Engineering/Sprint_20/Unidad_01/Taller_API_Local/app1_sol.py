import flask
from flask import request, jsonify
import sqlite3

app = flask.Flask(__name__)
app.config["DEBUG"] = True

books = [
    {'id': 0,
    'title': 'A Fire Upon the Deep',
    'author': 'Vernor Vinge',
    'first_sentence': 'The coldsleep itself was dreamless.',
    'year_published': '1992'},
    {'id': 1,
    'title': 'The Ones Who Walk Away From Omelas',
    'author': 'Ursula K. Le Guin',
    'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
    'published': '1973'},
    {'id': 2,
    'title': 'Dhalgren',
    'author': 'Samuel R. Delany',
    'first_sentence': 'to wound the autumnal city.',
    'published': '1975'},
    {'id': 3,
    'title': 'The Chain',
    'author': 'Jaime G. Páramo',
    'first_sentence': 'There were tears on her eyes and fears trapped her mind but, inside, the courage of those who have nothing to lose and all to win, flown wild and free.',
    'published': '2025'}
    ]

@app.route('/', methods=['GET'])
def home():
    return """
    <h1>Distant Reading Archive</h1>
    <p>This site is a prototype API for distant reading of science fiction novels.</p>
    """

@app.route('/api/v1/resources/books/all', methods=['GET'])
def api_all():
    return jsonify(books)

@app.route('/api/v1/resources/books', methods=['GET'])
def api_id():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."
    
    results = []

    for book in books:
        if book['id'] == id:
            results.append(book)
    return jsonify(results)


@app.route('/api/v1/resources/books/<string:title>', methods=['GET'])
def get_by_title(title):
    for book in books:
        if book['title'] == title:
            return jsonify(book)
    return jsonify({'message': "Book not found"})


@app.route('/api/v1/resources/books', methods=['POST'])
def post_book():
    data = request.get_json()
    books.append(data)
    return data


app.run()