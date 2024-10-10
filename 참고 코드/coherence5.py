# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import random
from bertopic import BERTopic
from transformers import BertTokenizer, BertModel
import logging
import time
import json

# 랜덤 시드 고정
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터 로드 함수
def load_data(file_path, sample_size=80):
    try:
        df = pd.read_csv(file_path, header=None, names=['text'])
        texts = df['text'].astype(str)
        if len(texts) > sample_size:
            texts = texts.sample(n=sample_size, random_state=42)
        logging.info(f"Loaded {len(texts)} texts from {file_path}")
        return texts.tolist()
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        return []

def load_all_datasets():
    datasets = {
        'academy': {
            'business': load_data('data/academy/business.csv'),
            'ACL': load_data('data/academy/ACL.csv'),
            'covid': load_data('data/academy/covid.csv')
        },
        'media': {
            'clothing_review': load_data('data/media/clothing_review.csv'),
            'vaccine_tweets': load_data('data/media/vaccine_tweets.csv'),
            'reddit_comments': load_data('data/media/reddit_comments.csv')
        },
        'news': {
            'newsgroups': load_data('data/news/20newsgroups.csv'),
            'agnews': load_data('data/news/agnews.csv'),
            'Huffpost': load_data('data/news/Huffpost.csv')
        }
    }
    return datasets

# VAE 모델 정의
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, latent_dim=10):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.log_softmax(self.fc4(h3), dim=1)  # log_softmax 적용

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# VAE 손실 함수 정의
def vae_loss(recon_x, x, mu, logvar):
    NLL = F.nll_loss(recon_x, x.argmax(dim=1), reduction='sum')  # NLLLoss 사용
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return NLL + KLD

# VAE 토픽 추출 함수 정의
def extract_vae_topics(vae_model, vectorizer, num_topics, top_n=10):
    decoder_weight = vae_model.fc4.weight.detach().cpu().numpy().T  # 전치 적용
    feature_names = vectorizer.get_feature_names_out()
    
    topics = []
    for topic_idx in range(num_topics):
        top_feature_indices = decoder_weight[topic_idx].argsort()[-top_n:][::-1]
        topic_words = [feature_names[i] for i in top_feature_indices]
        topics.append(topic_words)
    
    return topics

# VAE 모델 학습 함수 정의
def perform_vae_topic_modeling(data, num_epochs=5, hidden_dim=50, latent_dim=10):
    data = [str(doc) for doc in data if isinstance(doc, str) and len(doc) > 0]

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(data)
    vocab_size = len(vectorizer.get_feature_names_out())
    input_dim = vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

    batch_size = 64
    data_loader = DataLoader(doc_term_matrix.toarray(), batch_size=batch_size, shuffle=True)

    vae_model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch in data_loader:
            batch = torch.FloatTensor(batch).to(device)
            batch = batch / (batch.sum(dim=1, keepdim=True) + 1e-6)  # 정규화 수정
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae_model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        logging.info(f"에폭 {epoch+1}/{num_epochs}, 손실: {train_loss / len(data_loader.dataset):.4f}")

    topics = extract_vae_topics(vae_model, vectorizer, latent_dim)
    return vae_model, topics

# BERTopic 모델 학습 함수 정의
def perform_bertopic_modeling(data):
    data = [doc for doc in data if isinstance(doc, str) and len(doc) > 0]

    bertopic_model = BERTopic()
    topics, _ = bertopic_model.fit_transform(data)

    num_topics = len(set(topics)) - 1
    logging.info(f"추출된 토픽의 수: {num_topics}")

    topics_info = bertopic_model.get_topics()
    topics = []
    for topic_id in topics_info:
        if topic_id == -1:
            continue
        topic_words = [word for word, _ in topics_info[topic_id]][:10]  # 상위 10개 단어만 추출
        topics.append(topic_words)

    return bertopic_model, topics

# 새로운 Coherence 지표 계산 함수 정의
def calculate_coherence(topics, tokenizer, bert_model):
    coherence_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bert_model.to(device)
    bert_model.eval()
    
    for topic_words in topics:
        sentences = [f"The topic is about {word}" for word in topic_words]
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        num_words = len(topic_words)
        if num_words < 2:
            coherence_scores.append(0)
            continue
        
        pairwise_similarities = []
        for i in range(num_words):
            for j in range(i + 1, num_words):
                cosine_sim = torch.nn.functional.cosine_similarity(embeddings[i], embeddings[j], dim=0)
                pairwise_similarities.append(cosine_sim.item())
        
        coherence = np.mean(pairwise_similarities)
        coherence_scores.append(coherence)
    
    final_coherence = np.mean(coherence_scores) if coherence_scores else 0
    return final_coherence

def calculate_npmi(topics, tokenized_data, dictionary):
    coherence_model_npmi = CoherenceModel(topics=topics, texts=tokenized_data, dictionary=dictionary, coherence='c_npmi')
    npmi = coherence_model_npmi.get_coherence()
    return npmi

def calculate_umass(topics, corpus, dictionary):
    coherence_model_umass = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    umass = coherence_model_umass.get_coherence()
    return umass

def process_dataset(domain, dataset_name, data, model_types, num_topics_list, metrics_list, tokenizer, bert_model):
    for model_type in model_types:
        logging.info(f"Processing {domain} - {dataset_name} - {model_type}")
        try:
            if model_type == 'BERTopic':
                model, topics = perform_bertopic_modeling(data)
                num_topics = len(topics)
                process_metrics(domain, dataset_name, model_type, num_topics, topics, data, metrics_list, tokenizer, bert_model)
            elif model_type == 'VAE':
                for num_topics in num_topics_list:
                    model, topics = perform_vae_topic_modeling(data, latent_dim=num_topics)
                    process_metrics(domain, dataset_name, model_type, num_topics, topics, data, metrics_list, tokenizer, bert_model)
            else:
                continue
        except Exception as e:
            logging.error(f"Error processing {domain} - {dataset_name} - {model_type}: {str(e)}")
            continue

    return metrics_list

def process_metrics(domain, dataset_name, model_type, num_topics, topics, data, metrics_list, tokenizer, bert_model):
    tokenized_data = [simple_preprocess(doc) for doc in data]
    dictionary = Dictionary(tokenized_data)
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]

    coherence = calculate_coherence(topics, tokenizer, bert_model)
    npmi = calculate_npmi(topics, tokenized_data, dictionary)
    umass = calculate_umass(topics, corpus, dictionary)

    metrics_list.append({
        'Domain': domain,
        'Dataset': dataset_name,
        'Model': model_type,
        'Num_Topics': num_topics,
        'Coherence': coherence,
        'NPMI': npmi,
        'U_Mass': umass
    })

    logging.info(f"Coherence: {coherence:.4f}, NPMI: {npmi:.4f}, U_Mass: {umass:.4f}")

# 메인 실행 부분
if __name__ == '__main__':
    datasets = load_all_datasets()
    metrics_list = []
    computation_times = {}
    model_types = ['BERTopic', 'VAE']
    num_topics_list = [10]  # VAE에서 사용할 토픽 수

    # BERT 모델 및 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    for domain, domain_datasets in datasets.items():
        for dataset_name, data in domain_datasets.items():
            metrics_list = process_dataset(domain, dataset_name, data, model_types, num_topics_list, metrics_list, tokenizer, bert_model)

    # 결과를 DataFrame으로 변환하고 CSV로 저장
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('topic_modeling_metrics.csv', index=False)
    logging.info("Metrics saved to topic_modeling_metrics.csv")

    # 계산 시간을 JSON 파일로 저장
    with open('computation_times.json', 'w') as f:
        json.dump(computation_times, f)
    logging.info("Computation times saved to computation_times.json")

    # 결과 분석 및 시각화
    logging.info("\n=== 결과 분석 ===")
    avg_ranks = compare_average_ranks(metrics_df)
    agreement_results = analyze_agreement(metrics_df)
    stability_df = analyze_stability(datasets, model_types, num_topics_list)
    visualize_topic_quality(metrics_df)