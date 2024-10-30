import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def classify():
    num_trn, num_tst = 700, 300

    data = pd.read_csv(f"fake_and_real_news.csv")
    data.dropna(inplace=True)


    # Separating data from main dataset
    train = data.sample(n=num_trn)
    test = data.sample(n=num_tst)

    # Printing the data
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}\n")

    print(f"Train data: \n{train.head()}\n")
    print(f"Test data: \n{test.head()}")
    
    x_train, y_train = train["Text"], train["label"]
    x_test, y_test = test["Text"], test["label"]

    # Previewing the data
    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")

    print(f"x_train: {x_train.head()}")
    print(f"y_train: {y_train.head()}")
    print(f"x_test: {x_test.head()}")
    print(f"y_test: {y_test.head()}")
    
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

    # Fit and transform the training data
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    
    # initialize model
    model = LogisticRegression()

    # Fit the model
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
    
    y_pred = model.predict(x_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

    return vectorizer, model

def predict(title, vectorizer: TfidfVectorizer, model: LogisticRegression):
    input_text = [title]
    input_text = vectorizer.transform(input_text)
    prediction = model.predict(input_text)
    print(f"Prediction: {prediction}")

    return prediction