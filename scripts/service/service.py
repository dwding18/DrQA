from flask import Flask
from flask import jsonify
from flask import request
app = Flask(__name__)

import torch
import argparse
import logging
import json
from drqa import pipeline

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--reader-model', type=str, default=None,
                    help='Path to trained Document Reader model')
parser.add_argument('--retriever-model', type=str, default=None,
                    help='Path to Document Retriever model (tfidf)')
parser.add_argument('--doc-db', type=str, default=None,
                    help='Path to Document DB')
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to "
                          "use (e.g. 'corenlp')"))
args = parser.parse_args()

logger.info('Initializing pipeline...')
DrQA = pipeline.DrQA(
    cuda=False,
    fixed_candidates=None,
    reader_model=args.reader_model,
    ranker_config={'options': {'tfidf_path': args.retriever_model}},
    db_config={'options': {'db_path': args.doc_db}},
    tokenizer=None
)


@app.route("/")
def process():
    question = request.args.get('q', '')
    if not question:
        return 'hello bot'

    top_n = 3
    n_docs=5
    candidates = None
    predictions = DrQA.process(
        question, candidates, top_n, n_docs, return_context=True
    )

    result = []
    for i, p in enumerate(predictions, 1):
        result.append({ 'rank': i, 'span': p['span'], 'doc_id': p['doc_id'],
                        'span_score': p['span_score'], 
                        'doc_score': p['doc_score'],
                        'context': p['context']['text'] })

    return jsonify(result) 

if __name__ == '__main__':
   app.run()
