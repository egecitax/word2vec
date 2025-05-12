# 🧠 Word2Vec from Scratch (NumPy Only)

Bu proje, Word2Vec algoritmasının temel mantığını Python ve NumPy kullanarak sıfırdan (scratch) implemente etmektedir.  
Kütüphanelerden bağımsız, sade ve eğitici bir uygulamadır.

---

## 📌 Özellikler

- One-hot encoding
- Word-embedding matrisleri (W1 & W2)
- Softmax ve cross-entropy loss
- Manual forward pass
- Manual backward pass (gradient descent)
- Eğitim döngüsü (epoch + loss)
- NumPy dışında hiçbir kütüphane kullanılmadı

---

## 🧩 Model Yapısı

Input Word (One-hot)
↓
W1 (embedding)
↓
Hidden Layer (dense vector)
↓
W2 (projection)
↓
Softmax → Output Word (probability vector)


> Eğitim tamamlandığında, W1 matrisindeki her satır bir kelimenin **embedding vektörüdür**.

---

## 🚀 Kullanım

```bash
git clone https://github.com/kullanici_adi/word2vec-from-scratch.git
cd word2vec-from-scratch
python word2vec.py

