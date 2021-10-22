#import torch
#from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import math
import collections

import cherrypy
import sys
import os
import json
import requests


'''
This script includes two ways to call the results from the QA-for-Event_Extraction model:
1) Way 1: With HuggingFace API requested, send query to get back the results as a json file. 
   (See: https://api-inference.huggingface.co/docs/python/html/detailed_parameters.html#question-answering-task)
2) Way 2: Taking advantage of the HuggingFace Framework, download the pre-trained model in local and call the model to get the prediction. 
   Notice that the results returning from the model do not include confidence score. Therefore, We customized functions, 
   modified from squad_metrics.py (https://github.com/huggingface/transformers/blob/12b4d66a80419db30a15e7b9d4208ceb9887c03b/src/transformers/data/metrics/squad_metrics.py#L384),
   to calculate the confidence score and screen the best n predictions. Since the source code of the hosted API is in black box, 
   we have to decide some hyper-parameters by ourselves, which causes tiny difference between the results from the API and the cumstomized 
   functions. In addition, the prediction will be processed token, which means it returns lower-case tokens instead of the original tokens
   in the context.

In this script, the Way 1 is adapted for the consistence of the performance of the demo on Hugging Face. However, the second way is also 
kept for the convenience in case there is modification on the model in the future. 
'''

#--------------------------------- Way 1: Hosted inference API --------------------------------------
def inference_api(question, context, API_TOKEN='api_EZDSPRblVdvrKPAPRDQMOofeHwXBXpyLbS'):
    '''
    para:
        API_TOKEN can be copied from personal HuggingFace Profile. (https://api-inference.huggingface.co/docs/python/html/quicktour.html)
    return a dictionary like this:
    {"score": 0.9326569437980652, "start": 11, "end": 16, "answer": "Clara"}
    '''
    #API_TOKEN = 'api_EZDSPRblVdvrKPAPRDQMOofeHwXBXpyLbS'  
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/veronica320/QA-for-Event-Extraction"

    def query(payload):
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    try:
        data = query(
            {
                "inputs": {
                    "question":  question,#"What is the price of the ticket?",
                    "context":  context#"Barack Hussein Obama, an American politician serving as the 44th President of the United States, graduated from Columbia University and Harvard Law School, where he served as president of the Harvard Law Review."
                }
            }
        )

    except:
        raise RuntimeError("{'error': 'Model veronica320/QA-for-Event-Extraction is currently loading', 'estimated_time': 56.69663964}")

    return data

#--------------------------------- Way 2: Customized functions --------------------------------------


# def compute_logits(text, question):
#     '''
#     return the logits for start tokens, and the logits for the end tokens
#     '''
#     inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
#     input_ids = inputs["input_ids"].tolist()[0]
#     outputs = model(**inputs)
#     #print(f'outputs: {outputs}')
#     answer_start_scores = outputs.start_logits
#     answer_end_scores = outputs.end_logits
#     # # Get the most likely beginning of answer with the argmax of the score
#     answer_start = torch.argmax(answer_start_scores)
#     # Get the most likely end of answer with the argmax of the score
#     answer_end = torch.argmax(answer_end_scores) + 1
#     #print(answer_start, answer_end)
#     #answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
#     return input_ids, answer_start_scores, answer_end_scores

# ----------------- functions to calculate the confidence score and select the best valid prediction ----------------
#tokenizer = AutoTokenizer.from_pretrained("veronica320/QA-for-Event-Extraction")
#model = AutoModelForQuestionAnswering.from_pretrained("veronica320/QA-for-Event-Extraction")


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def compute_predictions_logits(question, context, n_best_size=10):
    ''' 
    return: the best valid prediction and its confidence score .
    '''
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    #print(f'outputs: {outputs}')
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    feature_len = len(start_logits[0]) # both start_logits and end_logits have the same length which is equal to the sum of the number of tokens in the question and context (with additional 4 paddings if add_special_tokens=True)
    start_indexes = _get_best_indexes(start_logits[0], n_best_size)
    end_indexes = _get_best_indexes(end_logits[0], n_best_size)

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )
    prelim_predictions = []

    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= feature_len:
                continue
            if end_index >= feature_len:
                continue
            # if start_index not in feature.token_to_orig_map:
            #     continue
            # if end_index not in feature.token_to_orig_map:
            #     continue
            # if not feature.token_is_max_context.get(start_index, False):
            #     continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            #if length > max_answer_length:
                #continue
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits[0][start_index],
                    end_logit=end_logits[0][end_index],
                )
            )

    prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        nbest.append(pred)


    total_scores = []
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)

    probs = _compute_softmax(total_scores)

    # get the best valid prediction:
    question_len = len(tokenizer(question, add_special_tokens=True, return_tensors="pt"))
    for index, best in enumerate(nbest):
        best_answer = 'None'
        confidence_score = 0
        if best.start_index > question_len:
            best_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[best.start_index:best.end_index+1]))
            confidence_score = probs[index]
            #print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[best.start_index:best.end_index])))
            #print(probs[index])
            break
        

    return best_answer, confidence_score#, probs, nbest





class Annotation(object):
    @cherrypy.expose
    def index(self):
         return open('./frontend/index.html')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def info(self, **params):
        return {"status":"online"}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def halt(self, **params):
        cherrypy.engine.exit()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def process(self, **params):
        # CherryPy passes all GET and POST variables as method parameters.
        # It doesn't make a difference where the variables come from, how
        # large their contents are, and so on.
        #
        # You can define default parameter values as usual. In this
        # example, the "name" parameter defaults to None so we can check
        # if a name was actually specified.
        
        try:
            data = cherrypy.request.json
            useJSON = True
            print("\nReading JSON Docs from Request")

        except:
            data = cherrypy.request.params
            print(data)
            useJSON = False
            print("\nReading Parameters from the URL")

        if useJSON:
            para = cherrypy.request.params
            if len(para) != 0:
                print("\nOverwrite JSON with Parameters (HTTP is priority)")
                data = para
        #print(data['context'])
        #pred, score = compute_predictions_logits(data['question'], data['context']) 
        pred_dic = inference_api(data['question'], data['context'])
        try:
            if pred_dic['score'] >= 0.1:
                pred = pred_dic['answer']
                score = pred_dic['score']
            else:
                pred = 'None'
                score = 'The confidence score is less than 0.1.'
            res = {}
            res['answer']= pred
            res['score']=score
            return res# pred, score
        except:
            res = 'The model is loading, please try it again in few seconds'
            return res


################################ Sys parameters ###############################
serviceURL = sys.argv[1]
servicePort = int(sys.argv[2])


if __name__ == '__main__':
    
    # A note that the service has started
    print("Starting rest service...")

    # A default configuration
    config = {'server.socket_host': serviceURL}
    cherrypy.config.update(config)

    # Update the configuration to your host
    cherrypy.config.update({'server.socket_port': servicePort})
    
    # cherrypy.config.update({'server.socket_host': 'dickens.seas.upenn.edu', 'server.socket_port': 4049})
    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
       '/js': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './frontend/js'
        },
       '/css': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './frontend/css'
        },
    }


    # Start the service
    cherrypy.quickstart(Annotation(), '/', conf)
