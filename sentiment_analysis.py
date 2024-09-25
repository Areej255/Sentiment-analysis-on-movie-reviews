# import libraries
import pandas as pd 

# load the dataset
train_data = pd.read_csv("C:\\Users\\PMLS\\Downloads\\train.tsv.zip",sep="\t")
test_data = pd.read_csv("C:\\Users\\PMLS\\Downloads\\test.tsv.zip",sep="\t")
# print(test_data.head(20))

# The nltk library is used for natural language processing tasks.
import nltk
# stopwords: This provides a list of common words (like "and", "the") that don't carry significant meaning.
from nltk.corpus import stopwords
# word_tokenize: This function breaks down text into individual words (tokens).
from nltk.tokenize import word_tokenize
# WordNetLemmatizer: This lemmatizer reduces words to their root form (e.g., "running" becomes "run").
from nltk.stem import WordNetLemmatizer


# download necessary nltk data
# nltk.download('punkt_tab')  # Download the 'punkt' tokenizer model
# nltk.download("stopwords")
# nltk.download("wordnet")

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# defining preprocessing func
def preprocess_text(text):
    # tokenize
    tokens = word_tokenize(text)
    # lowercase and remove stopwords
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    # lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

# # Applying Preprocessing to Data
# Apply the preprocess_text function to a few rows of the 'Phrase' column
train_data['cleaned_text'] = train_data['Phrase'].apply(preprocess_text)

# Check if 'cleaned_text' column was created
# print(train_data.columns)
# # Display the first few rows to verify the 'cleaned_text' column
# print(train_data.head())
# print(train_data['cleaned_text'].isnull().sum())  # Check for missing values

# # Show the first 5 rows of original and processed text
# print(train_data[['Phrase', 'cleaned_text']].head())


# Converting Text to Numerical Form (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Apply TF-IDF on the 'cleaned_text' column
X = tfidf.fit_transform(train_data['cleaned_text'])

# Target variable
y = train_data['Sentiment']

# Check shapes to verify successful transformation
# print(X.shape, y.shape)

# Splitting Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Training the Model (Logistic Regression or Naive Bayes)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler

# Scale the sparse data (works well with TF-IDF)
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the model
model = LogisticRegression(max_iter=300)
model.fit(X_train_scaled, y_train)
# (X_train as features, y_train as labels).
#  The model learns the relationship between the text features and their corresponding sentiments.

# uses the trained model to make predictions on the test dataset
y_predict = model.predict(X_test)

# model evaluation
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

# calculating metrics
# Computes the proportion of correctly predicted labels out of all test samples.
accuracy = accuracy_score(y_test,y_predict)

# Measures how many of the predicted positive sentiments were correct.
#  The weighted average accounts for label imbalance.
precision = precision_score(y_test,y_predict,average="weighted")

# Measures how many of the actual positive sentiments were predicted correctly.
recall = recall_score(y_test,y_predict,average="weighted")

# Combines precision and recall into a single score (F1-Score).
f1 = f1_score(y_test,y_predict,average="weighted")

# Displaying Classification Report
# print("Classification Report",classification_report(y_test,y_predict))

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=train_data["Sentiment"])
plt.title("Sentiment Distribution")
plt.show()