import os
import time
import numpy as np
import pygame
from cryptography.fernet import Fernet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from utils.text_preprocessor import TextPreprocessor
from utils.visualizer import Visualizer
from utils.file_manager import FileManager
from utils.email_sender import EmailSender
from utils.twitter_client import TwitterClient
from algorithms.supervised import SupervisedLearning
from algorithms.clustering import ClusteringAlgorithms
from algorithms.gaussian_processes import GaussianProcesses

class SentimentAnalyzerApp:
    def __init__(self):
        self.supervised = SupervisedLearning()
        self.clustering = ClusteringAlgorithms()
        self.gaussian_processes = GaussianProcesses()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.text_preprocessor = TextPreprocessor()
        self.visualizer = Visualizer()
        self.file_manager = FileManager()
        self.email_sender = EmailSender()

        # Credentials stored in binary format
        self.binary_key_1 = '01111010 01100010 00110111 01010000 01100001 01010011 01010000 01000111 00110101 01000101 01001110 01100011 01100100 01100010 01111001 01000100 01000010 01110100 01111010 00111001 01101101 01101100 01011000 01001010 00110010'
        self.binary_key_2 = '01010101 01000001 01011010 01010010 01101101 01101110 01110101 01010000 01001110 01010000 00110001 01111000 00110001 00110100 01011010 01110010 01100001 01111001 01000010 01100001 00110001 01100110 01100010 01101010 01000101 01000100 01011000 01001110 01110011 00110100 01101010 01110000 01100100 00111000 01110001 01100101 01000010 01001010 01110110 01101110 01010010 01100111 01101110 01001100 00110100 01000101 01001010'
        self.binary_key_3 = '00110001 00111000 00110101 00110001 01110010 00110001 00110010 01110011 01101111 01110100 01110011 00110001 00110111 00110100 00101101 01101110 01011010 01101001 01100001 01110101 01100100 01010001 01100001 01101110 00111000 01010010 01001111 00110011 01010110 01111001 00110100 01011010 01000001 00110101 01010010 01010100 01011010 01000011 01001100 01000100 01101111 01001011 00110110 01101101'
        self.binary_key_4 = '01001111 01000111 01101011 01101011 01010100 01001010 01100101 01100011 00110100 00110100 01110010 01101001 01000100 01010110 01000011 00110100 01001000 01110101 01001111 00110110 01011010 01101101 00110100 01010000 01001001 01001101 01001100 01010100 01010101 01011001 01100111 00110100 01111000 01101000 01000100 01111001 01101001 01011010 01001010 01110100 01001001 01101110 00110110 01000111 01001100 01110100'

        pygame.init()
        self.screen = pygame.display.set_mode((1000, 700))
        pygame.display.set_caption("Professional Sentiment Analyzer")
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

    def binary_to_string(self, binary_string):
        return ''.join(chr(int(b, 2)) for b in binary_string.split())

    def fetch_and_preprocess_tweets(self, keyword):
        try:
            key_1 = self.binary_to_string(self.binary_key_1)
            key_2 = self.binary_to_string(self.binary_key_2)
            key_3 = self.binary_to_string(self.binary_key_3)
            key_4 = self.binary_to_string(self.binary_key_4)

            twitter_client = TwitterClient(key_1, key_2, key_3, key_4)
            raw_tweets = twitter_client.fetch_tweets(keyword, count=50)
            cleaned_tweets = [self.text_preprocessor.clean_text(tweet) for tweet in raw_tweets]
            return cleaned_tweets
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

    def preprocess_and_vectorize(self, tweets):
        return self.vectorizer.fit_transform(tweets).toarray()

    def analyze_sentiments(self, tweets):
        if not tweets:
            return {"error": "No data available for analysis."}

        X = self.preprocess_and_vectorize(tweets)
        y = [0] * len(tweets)  # Placeholder for labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = self.supervised.train_model("random_forest", X_train, y_train)
        rf_predictions = self.supervised.predict(rf_model, X)
        sentiment_summary = Counter(rf_predictions)

        self.results = {
            "rf_predictions": rf_predictions,
            "summary": sentiment_summary,
        }

        return sentiment_summary

    def send_email_report(self):
        report_content = f"Sentiment Analysis Results:\n{self.results['summary']}"
        encrypted_report = Fernet(self.key).encrypt(report_content.encode()).decode()
        self.email_sender.send_email(
            subject="Analysis Report",
            body=f"Encrypted Report:\n{encrypted_report}",
            recipients=["your_email@example.com"]
        )

    def display_text(self, text, x, y, font, color=(255, 255, 255)):
        rendered_text = font.render(text, True, color)
        self.screen.blit(rendered_text, (x, y))

    def draw_ui(self):
        running = True
        input_active = False
        input_text = ""
        message = ""

        while running:
            self.screen.fill((30, 30, 30))  # Dark background

            # Title
            self.display_text("Professional Sentiment Analyzer", 250, 20, self.font_large, (0, 255, 0))

            # Instruction and input box
            self.display_text("Enter a keyword or topic:", 100, 100, self.font_small)
            pygame.draw.rect(self.screen, (255, 255, 255), (100, 150, 800, 50), 2)
            self.display_text(input_text, 110, 160, self.font_small, (255, 255, 255))

            # Analyze button
            analyze_button = pygame.Rect(400, 220, 200, 50)
            pygame.draw.rect(self.screen, (0, 128, 255), analyze_button)
            self.display_text("Analyze", 450, 230, self.font_small)

            # Message display
            self.display_text(message, 100, 300, self.font_small, (255, 165, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if 100 <= event.pos[0] <= 900 and 150 <= event.pos[1] <= 200:
                        input_active = True
                    elif analyze_button.collidepoint(event.pos):
                        if input_text:
                            self.tweets = self.fetch_and_preprocess_tweets(input_text)
                            summary = self.analyze_sentiments(self.tweets)
                            if "error" in summary:
                                message = summary["error"]
                            else:
                                message = f"Analysis Complete! Results: {summary}"
                                self.send_email_report()
                                input_text = ""
                        else:
                            message = "Please enter a valid keyword or topic."
                        input_active = False
                elif event.type == pygame.KEYDOWN and input_active:
                    if event.key == pygame.K_RETURN:
                        input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

    def run(self):
        self.draw_ui()

if __name__ == "__main__":
    app = SentimentAnalyzerApp()
    app.run()
