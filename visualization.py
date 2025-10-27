from model import *
from vocabulary import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib

class Visualization:
    def __init__(self,text_path,embedding_path):
        self.vocab=Vocabulary(text_path)

        model=CBOW(len(self.vocab),embedding_dim=10)
        model.load_embedding_weight(embedding_path)
        self.embeddings = model.embeddings.weight.data


        self.word_vectors = {}
        for word, idx in self.vocab.word2idx.items():
            if idx < self.embeddings.shape[0]: 
                self.word_vectors[word] = self.embeddings[idx]

       
    
    
    def find_similar_words(self,word,top_k=5):
        if word not in self.word_vectors:
            print(f"word '{word}' not included in vocabulary")
            return
    
        target_vector = self.word_vectors[word]
        similarities = []
    
        for match_word, vector in self.word_vectors.items():
            if match_word != word:
           
                cos_sim = torch.cosine_similarity(
                    target_vector.unsqueeze(0), 
                    vector.unsqueeze(0)
                )
                similarities.append((match_word, cos_sim.item()))
        
    
        similarities.sort(key=lambda x: x[1], reverse=True)
    
        print(f" {top_k} most similar words for '{word}':")
        for i, (similar_word, score) in enumerate(similarities[:top_k]):
            print(f"{i+1}. {similar_word}: {score:.4f}")

    def plot_word_vectors(self, words):
 
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
        matplotlib.rcParams['axes.unicode_minus'] = False  

        vectors = []
        labels = []

        for word in words:
            if word in self.word_vectors:
                vectors.append(self.word_vectors[word].numpy())
                labels.append(word)

        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)

        plt.figure(figsize=(10, 10))
        for i, label in enumerate(labels):
            x, y = reduced_vectors[i]
            plt.scatter(x, y)
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.title("Word Vectors PCA Projection")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid()
        plt.show()


