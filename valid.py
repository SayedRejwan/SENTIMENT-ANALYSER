from utils.environment_loader import EnvironmentLoader
from algorithms.supervised import SupervisedLearning
from algorithms.clustering import ClusteringAlgorithms
from algorithms.gaussian_processes import GaussianProcesses
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

def test_adaptive_learning():
    supervised = SupervisedLearning()
    vectorizer = TfidfVectorizer(max_features=1000)

    # Simulate real-time data for adaptive learning
    tweets = ["I am loving this feature!", "This tool is okay.", "Absolutely terrible experience."]
    labels = [1, 2, 0]  # Sentiment labels

    vectorized_data = vectorizer.fit_transform(tweets).toarray()
    supervised.train_model("random_forest", vectorized_data, labels)

    new_tweets = ["This is fantastic!", "Could be better."]
    new_labels = [1, 2]

    # Update the model adaptively
    vectorized_new_data = vectorizer.transform(new_tweets).toarray()
    supervised.update_model(vectorized_new_data, new_labels)
    print("Adaptive learning applied to the model.")

def test_outlier_detection():
    clustering = ClusteringAlgorithms()
    X = np.random.rand(100, 2)

    # Simulate outliers
    X[95:] = np.array([[10, 10], [12, 12], [9, 11], [10, 12], [11, 9]])

    labels = clustering.fit_dbscan(X)
    outliers = np.where(labels == -1)[0]

    print(f"Outliers detected at indices: {outliers}")

def test_explainable_ai():
    supervised = SupervisedLearning()
    vectorizer = TfidfVectorizer(max_features=1000)

    tweets = ["This feature is amazing!", "This is disappointing."]
    labels = [1, 0]

    vectorized_data = vectorizer.fit_transform(tweets).toarray()
    model = supervised.train_model("random_forest", vectorized_data, labels)

    explanations = supervised.explain_predictions(model, vectorized_data, vectorizer.get_feature_names_out())
    print("Explainable AI Results:")
    for i, explanation in enumerate(explanations):
        print(f"Tweet: {tweets[i]} => Explanation: {explanation}")

def test_multi_source_data_integration():
    print("Fetching data from multiple sources...")

    # Simulated data from Twitter
    twitter_data = ["Tweet about AI", "Another exciting tweet"]

    # Simulated data from Reddit
    reddit_data = ["Reddit post about AI", "Interesting discussion on Reddit"]

    combined_data = twitter_data + reddit_data
    print(f"Integrated data: {combined_data}")

def test_advanced_clustering():
    clustering = ClusteringAlgorithms()

    # Simulated hierarchical clustering data
    X = np.random.rand(50, 2)
    hierarchical_labels = clustering.fit_hierarchical(X)

    print(f"Hierarchical Cluster Labels: {hierarchical_labels}")

if __name__ == "__main__":
    print("Testing Adaptive Learning...")
    test_adaptive_learning()

    print("\nTesting Outlier Detection...")
    test_outlier_detection()

    print("\nTesting Explainable AI...")
    test_explainable_ai()

    print("\nTesting Multi-Source Data Integration...")
    test_multi_source_data_integration()

    print("\nTesting Advanced Clustering...")
    test_advanced_clustering()
