# **Sentiment Analyzer**

### **Overview**
The **Sentiment Analyzer** is a robust and interactive tool for real-time sentiment analysis of tweets. Using machine learning, data visualization, and secure credential management, it helps users understand public opinions, trends, and sentiment dynamics. With a modular design and an intuitive GUI, this project caters to researchers, businesses, and individuals alike.

---

## **Features**

### Real-Time Sentiment Analysis
- Fetch live tweets using the **Twitter API**.
- Classify tweets as **Positive**, **Negative**, or **Neutral** with a supervised learning model.
- Multi-language support with automatic translation of non-English tweets.

### Interactive GUI (Pygame)
- Dashboard-style interface for seamless interaction.
- Input fields, buttons, and real-time log display.
- Dark mode and user-friendly design.

### Advanced Visualizations
- **Pie Charts** for sentiment distribution.
- **Time-Series Plots** for trend analysis.
- Clustering insights using **K-Means** and **DBSCAN**.

### Automated Reporting
- Generate detailed **PDF reports** summarizing analysis results.
- Schedule daily or monthly reports sent via email.
- Summaries highlighting top positive, negative, and neutral tweets.

### Secure API Key Management
- Credentials are encrypted using the **cryptography** library.
- `.env` files for secure environment variable handling.

### Machine Learning Features
- Supervised sentiment analysis using **Random Forest**.
- Clustering algorithms for topic grouping.
- Gaussian Processes for regression and uncertainty estimation.

### Real-Time Monitoring
- Automated fetching and analysis of trending topics.
- Keyword comparison support.

---

## **Installation**

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Install Dependencies
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer
```

Install required Python libraries:
```bash
pip install -r requirements.txt
```

### Setup API Keys
1. Create a `.env` file in the root directory.
2. Add encrypted API credentials:
   ```env
   ENCRYPTED_API_KEY=b'encrypted_api_key_here'
   ENCRYPTED_API_SECRET=b'encrypted_api_secret_here'
   ENCRYPTED_ACCESS_TOKEN=b'encrypted_access_token_here'
   ENCRYPTED_ACCESS_TOKEN_SECRET=b'encrypted_access_token_secret_here'
   ENCRYPTION_KEY=b'encryption_key_here'
   ```

---

## **Usage**

### Run the Application
Start the application with:
```bash
python app.py
```

### Features in the GUI
- Enter a keyword to fetch and analyze tweets.
- View real-time sentiment results as pie charts and time-series plots.
- Generate a PDF report summarizing the analysis.

---

## **File Structure**

```
sentiment_project/
├── app.py                       # Main application entry point
├── valid.py                     # Validation script
├── requirements.txt             # Python dependencies
├── .env                         # Encrypted API credentials
├── utils/
│   ├── environment_loader.py    # Secure credential management
│   ├── text_preprocessor.py     # Text cleaning and preprocessing
│   ├── twitter_client.py        # Fetch tweets from Twitter API
│   ├── visualizer.py            # Data visualization tools
│   └── report_generator.py      # PDF report generation
├── algorithms/
│   ├── supervised.py            # Sentiment analysis (Random Forest)
│   ├── clustering.py            # K-Means and DBSCAN clustering
│   └── gaussian_processes.py    # Gaussian processes for regression
```

---

## **Contributing**

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
- **Tweepy** for Twitter API integration.
- **Scikit-Learn** for machine learning capabilities.
- **Matplotlib** for data visualization.
- **Cryptography** for secure API key encryption.
- **FPDF** for PDF report generation.
