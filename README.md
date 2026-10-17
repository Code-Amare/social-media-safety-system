SafeSpace — AI-Assisted Digital Wellbeing and Child Online Safety

About the Project

SafeSpace is an AI-powered project. It focuses on two important problems: unhealthy digital habits and children’s online safety.

I built this project to explore how Artificial Intelligence (AI) and Machine Learning (ML) can help people develop healthier relationships with technology while helping parents and guardians protect children from potentially inappropriate online content.

SafeSpace combines two AI systems into one project: a digital wellbeing risk prediction system and an AI-powered child online safety monitor.

Features

1. Digital Wellbeing Risk Prediction

This feature uses a Machine Learning model built with Python and XGBoost to estimate the risk of problematic digital use based on behavioral and lifestyle information.

The model uses user-provided information, such as screen time and other digital habits, to estimate the risk of problematic use.

The goal is to help users become more aware of their digital habits and recognize patterns that may negatively affect their daily lives.

Note: This feature is an experimental risk-estimation tool, not a medical diagnosis.

2. AI-Powered Child Online Safety

SafeSpace uses a pretrained Hugging Face image classification model, Falconsai/nsfw_image_detection, to identify potentially inappropriate visual content in screenshots.

Key features include:

* Capturing screenshots for content analysis.
* Analyzing images using a pretrained AI model.
* Identifying potentially inappropriate visual content.
* Sending Telegram notifications to a designated parent or guardian when configured detection criteria are met.

The goal is to help parents and guardians become more aware of potentially unsafe content children may encounter online.

3. Telegram Alerts

SafeSpace integrates with the Telegram Bot API to send notifications to a designated guardian when the system detects content that meets its configured alert criteria.

This feature aims to make the monitoring process more practical by notifying guardians without requiring them to watch the application continuously.

Technologies Used

* Python — main programming language
* XGBoost — digital wellbeing risk prediction
* Pandas — data processing
* Scikit-learn — machine learning utilities
* Hugging Face Transformers — pretrained image classification
* Streamlit — interactive dashboard, if enabled in the application
* Pillow / ImageGrab — image processing and screenshot capture, depending on the implementation
* Telegram Bot API — notifications
* Pickle — saving and loading the trained model pipeline

How SafeSpace Works

SafeSpace contains two main AI workflows.

Digital Wellbeing

User-provided digital habits → Data processing → XGBoost model → Estimated risk level

Child Online Safety

Screenshot capture → AI image classification → Content assessment → Telegram notification when configured criteria are met

Each workflow addresses a different aspect of digital wellbeing and online safety.

Project Structure

Example project structure (adjust the filenames to match your repository):

SafeSpace/
├── app.py
├── m.csv
├── addiction_pipeline.pkl
├── requirements.txt
└── README.md

* app.py — application entry point, if applicable.
* m.csv — dataset used to train the digital wellbeing model.
* addiction_pipeline.pkl — saved machine learning pipeline, if used.
* requirements.txt — required Python libraries.
* README.md — project documentation.

Getting Started

Prerequisites

* Python version compatible with the project’s dependencies.
* pip package manager.
* Required project files and trained model artifacts.
* A Telegram bot token if Telegram notifications are enabled.

Installation

Clone the repository:

git clone https://github.com/YOUR-USERNAME/SafeSpace.git

Enter the project directory:

cd SafeSpace

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

If your application uses Streamlit and the entry point is app.py, run:

streamlit run app.py

Replace the example repository URL and startup command with the ones that match your actual project.

Current Limitations

* Digital wellbeing predictions depend on the training data and the information provided by the user.
* The model cannot account for every factor that influences digital habits.
* Image classification can produce false positives or miss inappropriate content.
* Screenshot monitoring depends on operating-system permissions and the implementation.
* Telegram notifications require correct configuration and network access.
* More testing is needed to evaluate the system’s accuracy, reliability, and practical usefulness.

Future Improvements

I plan to improve SafeSpace by:

* Exploring automatic collection of screen-usage information with appropriate permission.
* Testing the models on more diverse datasets.
* Improving prediction reliability and reducing false alarms.
* Strengthening data security and privacy protection.
* Making the application easier for individuals and families to use.
* Evaluating its performance over longer periods.

Privacy and Responsible Use

SafeSpace may process sensitive information, including digital habits and screenshots. It should be used transparently, with appropriate consent and permissions. Personal information must be protected, and screenshot monitoring should respect privacy.

SafeSpace is an educational project. Its predictions and content classifications are not guaranteed to be accurate and should not be treated as a substitute for human judgment.

Acknowledgment

SafeSpace grew out of my interest in Artificial Intelligence, Machine Learning, digital wellbeing, and online safety during my experience at the EGATE Talent Center summer camp.

My goal is to continue developing AI tools that help people build healthier digital habits and create safer online experiences for children and families.

⸻

SafeSpace — Building healthier digital habits and safer online experiences through AI.
