# British Airways Review Analysis 🛫 📊

![Python Badge](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)
![Pandas Badge](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Numpy Badge](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![BeautifulSoup Badge](https://img.shields.io/badge/BeautifulSoup4-4.9.3-00979D?style=for-the-badge)
![NLTK Badge](https://img.shields.io/badge/NLTK-3.6.3-FF6F00?style=for-the-badge)
![TextBlob Badge](https://img.shields.io/badge/TextBlob-0.15.3-007ACC?style=for-the-badge)
![Scikit Badge](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![WordCloud Badge](https://img.shields.io/badge/WordCloud-1.8.1-4B8BBE?style=for-the-badge)
![Matplotlib Badge](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)
![Seaborn Badge](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge)
![Requests Badge](https://img.shields.io/badge/Requests-2.26.0-orange?style=for-the-badge)
![PyCharm Badge](https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white)
![Github Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)
![Markdown Badge](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)
![License Badge](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)


<div align="center">
  <img src="https://github.com/user-attachments/assets/bad7ba73-3b8a-406a-8ab9-3aa7d8126166" alt="image" width="600" height="400"/>
</div>



## Overview
This project analyzes customer sentiment and feedback from British Airways reviews using web scraping and machine learning techniques. By extracting insights from Skytrax reviews, we aim to understand customer satisfaction patterns and identify areas for service improvement.

## 🎯 Project Objectives:
- Extract customer reviews from Skytrax using web scraping
- Analyze sentiment patterns in customer feedback
- Identify key topics and themes in reviews
- Generate actionable insights through data visualization
- Build a comprehensive analysis pipeline for airline reviews

## 🚀 Dataset
The data is scraped from [Skytrax British Airways Reviews](https://www.airlinequality.com/airline-reviews/british-airways) and includes:

Key Features:
- Review Text: Customer feedback and comments
- Review Date: When the review was posted
- Cleaned Text: Processed version of reviews
- Sentiment Scores: Calculated sentiment polarity
- Topics: Extracted main themes from reviews

## 🛠 Installation

1. Clone the Repository
```bash
git clone https://github.com/YourUsername/british-airways-analysis.git
```

2. Set Up Python Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🔍 Project Workflow

### 1. Data Collection
```python
def scrape_skytrax_reviews():
    base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
    pages = 10
    page_size = 100
    reviews = []
    
    for i in range(1, pages + 1):
        url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"
        # Scraping logic...
    return reviews
```

### 2. Data Preprocessing
```python
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text
```

### 3. Analysis Pipeline
```python
# Sentiment Analysis
reviews_df['sentiment'] = reviews_df['cleaned_text'].apply(analyze_sentiment)

# Topic Modeling
topics = perform_topic_modeling(reviews_df['cleaned_text'])

# Generate Word Cloud
wordcloud = WordCloud(width=800, height=400).generate(all_text)
```

## 📊 Analysis Results

### Sentiment Distribution
![Sentiment Analysis](https://github.com/Astrolaboratory/BritishAirways/blob/main/sentiment_distribution.png)

### Word Cloud
![Word Cloud](https://github.com/Astrolaboratory/BritishAirways/blob/main/wordcloud.png)

### Key Metrics
```python
Total Reviews Analyzed: 1000+
Average Sentiment Score: 0.235
Positive Reviews: 65%
Negative Reviews: 35%
```

## 🧠 Machine Learning Models

### Sentiment Analysis Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| TextBlob | 85% | 0.87 | 0.83 | 0.85 |
| VADER | 82% | 0.84 | 0.81 | 0.82 |

## 🔮 Future Enhancements
- [ ] Implement advanced NLP techniques
- [ ] Add real-time review scraping
- [ ] Create interactive dashboards
- [ ] Develop predictive models
- [ ] Build API for automated analysis

## 👨‍💻 Technologies Used
```python
dependencies = {
    'python': '>=3.8',
    'beautifulsoup4': '4.9.3',
    'pandas': 'latest',
    'nltk': 'latest',
    'scikit-learn': 'latest',
    'textblob': 'latest',
    'wordcloud': 'latest',
    'matplotlib': 'latest',
    'seaborn': 'latest'
}
```

## 📜 License

This project is licensed under the **[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)**. Feel free to share and adapt the project with appropriate attribution.

---

## 📞 Contact Information

Feel free to reach out with any questions, feedback, or collaboration ideas. I am always open to new opportunities and discussions! 💬

---

### 👨‍💻 Ketan N  
## 🔗 **Connect With Me**
- 📧 **Email:** [Astronix@protonmail.com](mailto:Astronix@protonmail.com)
- 🔗 **LinkedIn:** [linkedin.com/in/ketannirpagare/](https://www.linkedin.com/in/ketannirpagare/)
- 🌍 **Portfolio:** [Data Science Portfolio](https://www.datascienceportfol.io/Ketannirpagare)
- 📜 **Medium:** [medium.com/@ketan.nirpagare](https://medium.com/@ketan.nirpagare)
- 💻 **GitHub:** [github.com/Astrolaboratory](https://github.com/Astrolaboratory)

---

### 📍 Let's Connect:
- 💼 **Open to job opportunities and freelance projects**!
- 🌱 Always learning new technologies and exploring innovative ways to leverage data for real-world solutions.

---

## 🌟 Acknowledgments
- Skytrax for providing the review platform
- British Airways dataset contributors
- Open source community for the amazing tools

---
💡 **Note**: This project is for educational purposes and is not affiliated with British Airways.

## 📝 How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🔧 Requirements.txt
```txt
beautifulsoup4==4.9.3
pandas==1.3.3
nltk==3.6.3
scikit-learn==0.24.2
textblob==0.15.3
wordcloud==1.8.1
matplotlib==3.4.3
seaborn==0.11.2
requests==2.26.0
