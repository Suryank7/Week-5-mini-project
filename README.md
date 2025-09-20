🎬 Movie Review Sentiment Analysis
This project uses Deep Learning (ANN with TensorFlow/Keras) to classify movie reviews as either positive or negative. It demonstrates text preprocessing, embedding, neural networks, and model evaluation with key performance metrics.

📌 Objective
The goal is to build a model that can understand natural language text and determine the sentiment (positive/negative) of a given movie review.

📊 Dataset
Source: IMDB Movie Reviews Dataset (Kaggle)
Size: 50,000 reviews (25k positive, 25k negative)
Format: CSV with two columns – review (text) and sentiment (label).

🧹 Preprocessing Steps
Text Cleaning – lowercasing, removing HTML tags, punctuation, and stopwords.
Tokenization – converting text into tokens (words → numbers).
Padding/Truncating – making all sequences the same length.
Encoding Labels – mapping sentiment: positive → 1, negative → 0.

🏗️ Model Architecture

A simple Artificial Neural Network (ANN):
Embedding Layer (input_dim=10000, output_dim=128, input_length=500)  
Flatten()  
Dense(128, activation='relu')  
Dropout(0.3)  
Dense(64, activation='relu')  
Dense(1, activation='sigmoid')  

⚙️ Training Setup
Optimizer: Adam
Loss Function: Binary Crossentropy
Epochs: 10–15
Batch Size: 128

📈 Evaluation Metrics
Accuracy
Precision, Recall, F1-score
Confusion Matrix

✅ Example Confusion Matrix:

[[11200, 1380],
 [1240, 11180]]

🔮 Results & Conclusion
Achieved ~89–91% accuracy on test data.
Model performs well in distinguishing positive and negative reviews.
Some errors occur in sarcastic or ambiguous reviews, which is expected in sentiment analysis.

Conclusion:
This project demonstrates the power of deep learning for NLP tasks. With further improvements like LSTMs, GRUs, or pretrained embeddings (GloVe/FastText), performance can be pushed even higher.

🚀 Extensions (Future Work)
Use LSTM/GRU for better context understanding.
Apply pretrained embeddings (GloVe, Word2Vec, FastText).
Deploy as a Flask/Streamlit Web App.
Extend to multi-class sentiment analysis (e.g., very positive → very negative).

🛠️ Tools & Libraries
TensorFlow/Keras – Deep Learning
NumPy, Pandas – Data handling
Scikit-learn – Evaluation metrics
Matplotlib/Seaborn – Visualization
