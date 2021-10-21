# QA-idk-demo

This is a QA model as part of the event extraction system in the ACL2021 paper: [Zero-shot Event Extraction via Transfer Learning: Challenges and Insights](https://aclanthology.org/2021.acl-short.42/). The pretrained architecture is roberta-large and the fine-tuning data is QAMR.

Both the backend and the frontend of the demo are saved in this repository. 

In the QA-idk-demo directory, the demo can start by running in terminal:
```
sh run.sh 
```
or by running backend.py with specific server url and port, such as:
```
python backend.py 0.0.0.0 8081
```
Then open another terminal, and send the curl request. The request consists of "context" (the context of the question), "question" (the question want to ask the model). An example is:
```
curl -d '{"context": "My name is Sarah and I live in London.", "question": "Where do I live?"}' -H "Content-Type: application/json" -X POST http://localhost:8081/process/
```

The results also can be acquired by taking advantage of sending query to the HuggingFace Hosted API Inference:
```
import json
import requests

headers = {"Authorization": f"Bearer {API_TOKEN}"} #the API_TOKEN can be found in the personal Hugging Face profile.
API_URL = "https://api-inference.huggingface.co/models/veronica320/QA-for-Event-Extraction"

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


data = query(
    {
        "inputs": {
            "question":  "Who is president of the Harvard Law Review?",
            "context":  "Barack Hussein Obama, an American politician serving as the 44th President of the United States, graduated from Columbia University and Harvard Law School, where he served as president of the Harvard Law Review."
        }
    }
)
```


To download the pre-trained model and get more information, please go to the HuggingFace [host page](https://huggingface.co/veronica320/QA-for-Event-Extraction) 

