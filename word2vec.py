import numpy as np

corpus = ["Ege sizi seviyor",
          "Okula karşı nötrümdür",
          "Çiğköfte ve milkshake",
          "Leziz"]

words = set(" ".join(corpus).split())
unique = list(set(words))

word2idx = {word: idx for idx, word in enumerate(unique)}
idx2word = {idx: word for word, idx in word2idx.items()}

vocab_size = len(word2idx)
embedding_dim = 5

#print(word2idx)
#print(idx2word)

def doubleize(corpus,window_size = 1):
    trainingdata = []
    for sentence in corpus:
        words = sentence.split()
        for idx, word in enumerate(words):
            for neighbor in range(max(idx-window_size,0),min(idx+window_size+1,len(words))):
                if neighbor != idx:
                    trainingdata.append((word,words[neighbor]))
    return trainingdata

trainingdata = doubleize(corpus)

#print("training data: " +trainingdata)

def training(vocab_size,embedding_dim):
    W1 = np.random.rand(vocab_size,embedding_dim)*0.01
    W2 = np.random.rand(embedding_dim,vocab_size) * 0.01

    return W1,W2

def one_hot(word_idx,vocab_size):
    vec = np.zeros(vocab_size)
    vec[word_idx] = 1
    return vec

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def forward_pass(x,W1,W2):
    h = np.dot(W1.T,x)
    u = np.dot(W2.T,h)
    y_pred = softmax(u)
    return y_pred,h


def loss(y_true,y_pred):
    return -np.sum(y_true* np.log(y_pred+1e-7))

def backprop(y_pred, y_true, h, W2, x, W1, learning_rate):
    e = y_pred - y_true  # hata vektörü
    dW2 = np.outer(h, e)  # (embedding_dim, vocab_size)
    dW1 = np.outer(x, np.dot(W2, e))  # (vocab_size, embedding_dim)

    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2

    return W1, W2

def train(training_data,word2idx,W1,W2,vocab_size,learning_rate =0.01, epochs = 100):

    for epoch in range(epochs):
        total_loss = 0
        for input_word,output_word in training_data:
            input_idx = word2idx[input_word]
            output_idx = word2idx[output_word]

            x = one_hot(input_idx,vocab_size)
            y_true = one_hot(output_idx,vocab_size)

            y_pred, h = forward_pass(x,W1,W2)

            current_loss = loss(y_true,y_pred)
            total_loss += current_loss

            W1,W2 =backprop(y_pred,y_true,h,W2,x,W1,learning_rate)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    return W1,W2


W1,W2 = training(vocab_size,embedding_dim)
W1,W2 = train(trainingdata,word2idx,W1,W2,vocab_size)

print(W1[word2idx["Leziz"]])