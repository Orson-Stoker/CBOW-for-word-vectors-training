from collections import Counter
import torch

class Vocabulary:
    def __init__(self,text_path):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_count = Counter()
        self.sentences=[]
        with open(text_path,'r',encoding="utf-8") as file:
            for line in file:
                 sentence=line.strip()
                 self.word_count.update(sentence.split())
                 self.sentences.append(sentence)
        idx = 2
        for word, count in self.word_count.items():
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def __len__(self):
         return len(self.word2idx)      

    def encode(self,sentence):
        return [self.word2idx.get(word, 1) for word in sentence.split()]
    

    def cbow_data(self,window_size=2):      
        contexts=[]
        targets=[]

        for sentence in self.sentences:
            encoded = self.encode(sentence)

            for i in range(window_size, len(encoded) - window_size):
                context = encoded[i-window_size:i] + encoded[i+1:i+window_size+1]
                target = encoded[i]
 
                contexts.append(context)
                targets.append(target)
    
        return torch.tensor(contexts), torch.tensor(targets)

         
    

                



