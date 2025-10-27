from vocabulary import *
from model import *
import torch.optim as optim

def train(text_path,epochs=100,batch_size=32,learning_rate=0.01,gpu=True):
    
    vocab = Vocabulary(text_path)
    model=CBOW(vocab_size=len(vocab),embedding_dim=10)
    loss_fn=nn.CrossEntropyLoss()

    if gpu==True:
        model=model.cuda()
        loss_fn=loss_fn.cuda()


    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    contexts,targets=vocab.cbow_data()
    losses=[]

    for epoch in range(epochs):
        total_loss=0
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            if gpu==True:
                batch_contexts=batch_contexts.cuda()
                batch_targets=batch_targets.cuda()
   
            outputs = model(batch_contexts)
            loss = loss_fn(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss+=loss.item()

        
        avg_loss = total_loss / (len(contexts) // batch_size)
        losses.append(avg_loss)
        if epoch%10==0:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    torch.save(model.embeddings.state_dict(),"embeddings_zh_10.pth")
    return losses
