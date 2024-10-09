# 클로드에 의해 전체 코드 수정함

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim import models, corpora
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
import time
import json
from nltk.corpus import stopwords
from math import log
from itertools import combinations
from tqdm import tqdm
import logging
from collections import Counter
import gensim
from gensim import corpora
from scipy.sparse import csr_matrix
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from transformers import BertTokenizer, BertModel
from bertopic import BERTopic
import seaborn as sns
from scipy import stats
import openai
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

# NLTK 데이터 다운로드 (한 번만 실행)
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# stop_words 정의
stop_words = set(stopwords.words('english'))

def load_data(file_path, sample_size=80):
    df = pd.read_csv(file_path, header=None, names=['text'])
    texts = df['text'].astype(str)
    if len(texts) > sample_size:
        texts = texts.sample(n=sample_size, random_state=42)
    print(f"Loaded {len(texts)} texts from {file_path}")
    return texts.tolist()

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

def vae_loss(recon_x, x, mu, logvar):
    NLL = F.nll_loss(recon_x, x.argmax(dim=1), reduction='sum')  # NLLLoss 사용
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return NLL + KLD

def extract_vae_topics(vae_model, vectorizer, num_topics, top_n=10):
    decoder_weight = vae_model.fc4.weight.detach().cpu().numpy().T  # 전치 적용
    feature_names = vectorizer.get_feature_names_out()
    
    topics = []
    for topic_idx in range(num_topics):
        top_feature_indices = decoder_weight[topic_idx].argsort()[-top_n:][::-1]
        topic_words = [feature_names[i] for i in top_feature_indices]
        topics.append(topic_words)
    
    return topics

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

def calculate_npmi(topic_words_with_weights, corpus, dictionary, top_n=10):
    topic_words = [word for word, _ in topic_words_with_weights[:top_n] if isinstance(word, str)]
    
    if not topic_words:
        logging.warning("토픽 단어가 없습니다.")
        return float('nan')
    
    npmi_scores = []
    doc_count = len(corpus)

    for i, word1 in enumerate(topic_words):
        for j, word2 in enumerate(topic_words):
            if i < j:
                try:
                    word1_id = dictionary.token2id[word1]
                    word2_id = dictionary.token2id[word2]

                    doc_count_word1 = dictionary.dfs[word1_id]
                    doc_count_word2 = dictionary.dfs[word2_id]
                    co_doc_count = sum(1 for doc in corpus if word1_id in doc and word2_id in doc)

                    p_w1 = doc_count_word1 / doc_count
                    p_w2 = doc_count_word2 / doc_count
                    p_w1_w2 = co_doc_count / doc_count

                    pmi = np.log(p_w1_w2 / (p_w1 * p_w2) + 1e-10)
                    npmi = pmi / (-np.log(p_w1_w2 + 1e-10))
                    npmi_scores.append(npmi)
                except KeyError:
                    logging.warning(f"단어 '{word1}' 또는 '{word2}'이 사전에 없습니다.")
                    continue

    if not npmi_scores:
        logging.warning("NPMI 계산 중 유효한 쌍이 없음.")
        return float('nan')

    return np.mean(npmi_scores)

def calculate_umass(topic_words_with_weights, corpus, dictionary, top_n=10):
    topic_words = [word for word, _ in topic_words_with_weights[:top_n] if isinstance(word, str)]
    
    if not topic_words:
        return 0

    umass_scores = []
    for i, word1 in enumerate(topic_words):
        for j, word2 in enumerate(topic_words):
            if i < j:
                if word1 in dictionary.token2id and word2 in dictionary.token2id:
                    co_occurrence = dictionary.dfs[dictionary.token2id[word1]] + 1
                    word1_occurrence = dictionary.dfs[dictionary.token2id[word2]] + 1
                    umass = np.log((co_occurrence / word1_occurrence) + 1e-10)
                    umass_scores.append(umass)

    return np.mean(umass_scores) if umass_scores else 0

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 평균 순위 비교 함수
def compare_average_ranks(metrics_df):
    metrics = ['Coherence', 'NPMI', 'U_Mass']
    for metric in metrics:
        metrics_df[f'{metric}_rank'] = metrics_df.groupby(['Domain', 'Dataset', 'Model', 'Num_Topics'])[metric].rank(ascending=False)
    
    top_25_percent = metrics_df.groupby(['Domain', 'Dataset', 'Model', 'Num_Topics'])[metrics].transform(lambda x: x >= x.quantile(0.75))
    
    avg_ranks = {}
    for metric in metrics:
        avg_ranks[metric] = metrics_df[top_25_percent[metric]][f'{metric}_rank'].mean()
    
    print("평균 순위 비교 결과:")
    for metric, avg_rank in avg_ranks.items():
        print(f"{metric}: {avg_rank:.2f}")
    
    return avg_ranks

# 일치도 분석 함수 (계속)
def analyze_agreement(metrics_df):
    metrics = ['Coherence', 'NPMI', 'U_Mass']
    correlations = {}
    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            metric1, metric2 = metrics[i], metrics[j]
            spearman_corr, _ = stats.spearmanr(metrics_df[metric1], metrics_df[metric2])
            correlations[f'{metric1} vs {metric2}'] = spearman_corr
    
    print("\n일치도 분석 결과 (Spearman 상관계수):")
    for pair, corr in correlations.items():
        print(f"{pair}: {corr:.4f}")
    
    return correlations

# 안정성 분석 함수
def analyze_stability(datasets, model_types, num_topics_list, n_runs=10, sample_ratio=0.8):
    stability_results = []
    
    for domain, domain_datasets in datasets.items():
        for dataset_name, data in domain_datasets.items():
            for model_type in model_types:
                for num_topics in num_topics_list:
                    metric_values = {
                        'Coherence': [],
                        'NPMI': [],
                        'U_Mass': []
                    }
                    
                    for _ in range(n_runs):
                        sampled_data = np.random.choice(data, size=int(len(data) * sample_ratio), replace=False)
                        model, vectorizer, topics = perform_topic_modeling(sampled_data, num_topics, model_type)
                        
                        coherence = calculate_coherence(model, sampled_data, tokenizer, bert_model, model_type, vectorizer, num_topics, topics)
                        
                        tokenized_data = [simple_preprocess(doc) for doc in sampled_data]
                        dictionary = Dictionary(tokenized_data)
                        corpus = [dictionary.doc2bow(text) for text in tokenized_data]
                        
                        npmi = calculate_npmi(topics[0], corpus, dictionary)
                        umass = calculate_umass(topics[0], corpus, dictionary)
                        
                        metric_values['Coherence'].append(coherence)
                        metric_values['NPMI'].append(npmi)
                        metric_values['U_Mass'].append(umass)
                    
                    for metric, values in metric_values.items():
                        cv = np.std(values) / np.mean(values)
                        stability_results.append({
                            'Domain': domain,
                            'Dataset': dataset_name,
                            'Model': model_type,
                            'Num_Topics': num_topics,
                            'Metric': metric,
                            'CV': cv
                        })
    
    stability_df = pd.DataFrame(stability_results)
    print("\n안정성 분석 결과 (변동계수):")
    print(stability_df.groupby(['Metric', 'Model'])['CV'].mean().unstack())
    
    return stability_df

# 개선된 토픽 품질 시각화 함수
def visualize_topic_quality(metrics_df):
    # 다차원 척도법(MDS) 시각화
    mds = MDS(n_components=2, random_state=42)
    metrics = ['Coherence', 'NPMI', 'U_Mass']
    mds_coords = mds.fit_transform(metrics_df[metrics])
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(mds_coords[:, 0], mds_coords[:, 1], 
                          c=metrics_df['Coherence'], cmap='viridis', 
                          s=50, alpha=0.6)
    plt.colorbar(scatter, label='Coherence')
    plt.title('MDS Visualization of Topic Quality')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.savefig('mds_topic_quality.png')
    plt.close()

    # 상관관계 히트맵
    corr_matrix = metrics_df[metrics].corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Coherence Metrics')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # 지표별 토픽 순위 변화 그래프
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(range(len(metrics_df)), metrics_df[metric].rank(ascending=False), label=metric)
    plt.xlabel('Topics')
    plt.ylabel('Rank')
    plt.title('Topic Ranks by Different Metrics')
    plt.legend()
    plt.savefig('topic_ranks_comparison.png')
    plt.close()

    # 토픽 품질 분포 비교
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        sns.kdeplot(metrics_df[metric], label=metric)
    plt.xlabel('Metric Value')
    plt.ylabel('Density')
    plt.title('Distribution of Topic Quality Metrics')
    plt.legend()
    plt.savefig('topic_quality_distribution.png')
    plt.close()

# 메인 함수
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

    # 새로운 분석 방법 적용
    avg_ranks = compare_average_ranks(metrics_df)
    agreement_results = analyze_agreement(metrics_df)
    stability_df = analyze_stability(datasets, model_types, num_topics_list)
    visualize_topic_quality(metrics_df)

    # 결과 분석 및 시각화
    logging.info("\n=== 결과 분석 ===")
    
    # 모델별 평균 성능
    logging.info("\n모델별 평균 성능:")
    logging.info(metrics_df.groupby('Model')[['Coherence', 'NPMI', 'U_Mass']].mean())

    # 토픽 수별 평균 성능
    logging.info("\n토픽 수별 평균 성능:")
    logging.info(metrics_df.groupby('Num_Topics')[['Coherence', 'NPMI', 'U_Mass']].mean())

    # 도메인별 평균 성능
    logging.info("\n도메인별 평균 성능:")
    logging.info(metrics_df.groupby('Domain')[['Coherence', 'NPMI', 'U_Mass']].mean())

    # 상관관계 분석
    correlation_metrics = ['Coherence', 'NPMI', 'U_Mass']
    corr_matrix = metrics_df[correlation_metrics].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Coherence Metrics')
    plt.tight_layout()
    plt.savefig('coherence_correlation_heatmap.png')
    plt.close()

    logging.info("\n상관계수 매트릭스:")
    logging.info(corr_matrix)

    # 각 지표 쌍에 대한 상관관계 분석
    for i in range(len(correlation_metrics)):
        for j in range(i+1, len(correlation_metrics)):
            metric1 = correlation_metrics[i]
            metric2 = correlation_metrics[j]
            
            pearson_corr, p_value = stats.pearsonr(metrics_df[metric1], metrics_df[metric2])
            logging.info(f"\n{metric1}와 {metric2}의 Pearson 상관계수: {pearson_corr:.4f}")
            logging.info(f"p-value: {p_value:.4f}")

            spearman_corr, p_value = stats.spearmanr(metrics_df[metric1], metrics_df[metric2])
            logging.info(f"{metric1}와 {metric2}의 Spearman 순위 상관계수: {spearman_corr:.4f}")
            logging.info(f"p-value: {p_value:.4f}")

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=metric1, y=metric2, data=metrics_df)
            plt.title(f'{metric1} vs {metric2}')
            plt.tight_layout()
            plt.savefig(f'coherence_{metric1}_vs_{metric2}_scatter.png')
            plt.close()

    # 모델 유형별 분석
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y='Coherence', data=metrics_df)
    plt.title('모델별 Coherence 분포')
    plt.savefig('model_coherence_distribution.png')
    plt.close()

    for model in metrics_df['Model'].unique():
        model_df = metrics_df[metrics_df['Model'] == model]
        logging.info(f"\n모델 {model}에 대한 상관계수:")
        logging.info(model_df[correlation_metrics].corr())

    # 토픽 수별 분석
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Num_Topics', y='Coherence', hue='Model', data=metrics_df)
    plt.title('토픽 수에 따른 Coherence 변화')
    plt.savefig('topic_number_coherence_trend.png')
    plt.close()

    for num_topics in metrics_df['Num_Topics'].unique():
        topic_df = metrics_df[metrics_df['Num_Topics'] == num_topics]
        logging.info(f"\n토픽 수 {num_topics}에 대한 상관계수:")
        logging.info(topic_df[correlation_metrics].corr())

    # 도메인별 분석
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Domain', y='Coherence', data=metrics_df)
    plt.title('도메인별 Coherence 분포')
    plt.savefig('domain_coherence_distribution.png')
    plt.close()
    
    # 결과 요약 및 해석
    print("\n=== 결과 요약 및 해석 ===")
    print(f"1. 평균 순위 비교: {avg_ranks}")
    print("   - 낮은 값일수록 해당 지표가 '좋은 토픽'을 더 잘 식별함을 의미합니다.")

    print(f"\n2. 일치도 분석: {agreement_results}")
    print("   - 1에 가까울수록 두 지표 간 높은 일치도를 나타냅니다.")

    print("\n3. 안정성 분석:")
    print(stability_df.groupby(['Metric', 'Model'])['CV'].mean().unstack())
    print("   - 낮은 CV 값일수록 해당 지표가 더 안정적임을 나타냅니다.")

    print("\n4. 시각화 결과:")
    print("   - MDS 시각화, 상관관계 히트맵, 토픽 순위 변화 그래프, 토픽 품질 분포 비교 그래프가 생성되었습니다.")
    print("   - 이미지 파일을 확인하여 시각적 분석을 수행하세요.")

    print("\n5. 결론:")
    print("   - 새로운 Coherence 지표의 성능을 기존 NPMI, U_Mass와 비교하여 해석하세요.")
    print("   - 평균 순위, 일치도, 안정성, 분포 등을 종합적으로 고려하여 평가하세요.")
    print("   - 도메인, 모델 유형, 토픽 수에 따른 차이도 고려하세요.")

    logging.info("분석 완료. 결과를 확인하고 해석하세요.")