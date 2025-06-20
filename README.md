# simple-RAG
A simple RAG application, you can use it to answer questions and chat with a PDF file.

## 🛠 Features
- Chat with PDF
- Accurate answers

## AI Providers
- Cohere
- Groq

## 📋 Requirements
- Python 3.11

## 🚀 Quickstart

1- Fork and Clone repo
```bash
 https://github.com/24-mohamedyehia/simple-RAG.git
```

2- 📦 Install Python Using Miniconda
 - Download and install MiniConda from [here](https://www.anaconda.com/docs/getting-started/miniconda/main#quick-command-line-install)

3- Create a new environment using the following command:
```bash
$ conda create --name simple-RAG python=3.11 -y
```

4- Activate the environment:
```bash
$ conda activate simple-RAG
```

5- Install the required packages
```bash
$ pip install -r requirements.txt
```

6- Setup the environment variables
```bash
$ cp .env.example .env
```

7- Set your environment variables in the .env file. Like:
- COHERE_API_KEY value to use LLM
    - You can get your Cohere API key from [here](https://dashboard.cohere.com/api-keys).
- GROQ_API_KEY value to monitor the agents
    - You can get your Groq API key from [here](https://console.groq.com/keys).

### 🚀 Run the application
```bash
$ python main.py
```

## 🛠 Technologies
- Python 3.11
- LangChain
- NLTK
- Chromadb

## 📜 License
This project is licensed under the MIT License See the [LICENSE](./LICENSE) file for details.