import torch
import json
from neural_model import NeuralNet
from intent_classes import Pattern, Tag
from nltk_utils import NLP_Util
import numpy as np


FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

class ChatAI:
    no_answer = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bot_name = "CCSA-Bot"
    
    def __init__(self):
        self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()
        
    def response(self, msg, rephrase_repeat = True) -> str:
        pattern = Pattern(msg, record_stem = False)
        np_pattern = np.array(pattern.in_bag_words)
        np_pattern = np_pattern.reshape(1, np_pattern.shape[0])
        tensor_pattern = torch.from_numpy(np_pattern).to(self.device)
        
        output = self.model(tensor_pattern)
        _, predicted = torch.max(output, dim=1)
        
        tag = Tag.get_tag(int(predicted.item()))
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            return tag.execute_response()
        return self.idunno_response(msg)
    
    def rephrase_input(self, pattern: Pattern) -> str:
        msg = str(pattern)
        old_msg = msg
        for token in pattern.tokenize:
            synonyms = NLP_Util.find_synonyms(token, all_words)
            if len(synonyms) > 0:
                msg = msg.replace(token, synonyms[0])
        if old_msg != msg:
            return self.response(msg, False)
        return self.idunno_response(msg)
    
    def idunno_response(self, msg_input: str) -> str:
        with open("No Answer Inputs.txt", "a") as txtfile:
            txtfile.write("-> " + msg_input)
        return "I don't Understand"
    
    
if __name__ == "__main__":
    Chat = ChatAI()
    print(f"Hi I'm {ChatAI.bot_name}, Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = Chat.response(sentence)
        print(resp)