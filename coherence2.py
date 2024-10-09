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

# NLTK 데이터 다운로드 (한 번만 실행)
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# stop_words 정의
stop_words = set(stopwords.words('english'))

def load_data(file_path, sample_size=100):
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
    def __init__(self, input_dim, hidden_dim, latent_dim):
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
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def perform_topic_modeling(data, num_topics, model_type):
    data = [str(doc) for doc in data if isinstance(doc, str) or pd.notna(doc)]
    
    if num_topics > len(data):
        print(f"Adjusting num_topics from {num_topics} to {len(data)}")
        num_topics = len(data)

    vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(data)

    if model_type == 'LDA':
        corpus = gensim.matutils.Sparse2Corpus(csr_matrix(doc_term_matrix))
        id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())
        lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics, workers=2, passes=10, random_state=42)
        topics = [lda_model.show_topic(i, topn=10) for i in range(num_topics)]
        return lda_model, vectorizer, topics

    elif model_type == 'BERTopic':
        bertopic_model = BERTopic(nr_topics=num_topics)
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        bertopic_topics, _ = bertopic_model.fit_transform(data)
        topics = [bertopic_model.get_topic(i) for i in range(num_topics) if bertopic_model.get_topic(i)]
        return bertopic_model, None, topics

    elif model_type == 'VAE':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = doc_term_matrix.shape[1]
        hidden_dim = 256
        latent_dim = num_topics

        vae_model = VAE(input_dim, hidden_dim, latent_dim).to(device)
        optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

        num_epochs = 50
        batch_size = 128

        for epoch in range(num_epochs):
            for i in range(0, doc_term_matrix.shape[0], batch_size):
                batch = torch.FloatTensor(doc_term_matrix[i:i+batch_size].toarray()).to(device)
                batch = batch / batch.max()
                optimizer.zero_grad()
                recon_batch, mu, logvar = vae_model(batch)
                loss = vae_loss(recon_batch, batch, mu, logvar)
                loss.backward()
                optimizer.step()

        topics = extract_vae_topics(vae_model, vectorizer, num_topics)
        return vae_model, vectorizer, topics

    else:
        raise ValueError("Invalid model type")

def extract_vae_topics(model, vectorizer, num_topics):
    feature_names = vectorizer.get_feature_names_out()
    weight = model.fc4.weight.data.cpu().numpy()
    topics = []
    for i in range(num_topics):
        top_words = [feature_names[j] for j in weight[i].argsort()[-10:][::-1]]
        topics.append([(word, weight[i][j]) for j, word in enumerate(top_words)])
    return topics

def calculate_npmi(topic_words_with_weights, texts, top_n=10):
    topic_words = [word for word, _ in topic_words_with_weights[:top_n] if isinstance(word, str)]
    
    if not topic_words:
        return 0

    vectorizer = CountVectorizer(vocabulary=topic_words)
    doc_word_matrix = vectorizer.fit_transform(texts)
    
    word_doc_counts = doc_word_matrix.sum(axis=0).A1
    doc_count = len(texts)
    
    npmi_scores = []
    for i, word1 in enumerate(topic_words):
        for j, word2 in enumerate(topic_words):
            if i < j:
                idx1 = vectorizer.vocabulary_[word1]
                idx2 = vectorizer.vocabulary_[word2]
                co_doc_count = doc_word_matrix[:, idx1].multiply(doc_word_matrix[:, idx2]).nnz
                if co_doc_count == 0:
                    continue

                p_w1 = word_doc_counts[idx1] / doc_count
                p_w2 = word_doc_counts[idx2] / doc_count
                p_w1_w2 = co_doc_count / doc_count

                pmi = np.log(p_w1_w2 / (p_w1 * p_w2) + 1e-10)
                npmi = pmi / (-np.log(p_w1_w2 + 1e-10))
                npmi_scores.append(npmi)

    return np.mean(npmi_scores) if npmi_scores else 0

def calculate_umass(topic_words_with_weights, corpus, dictionary, top_n=10):
    topic_words = [word for word, _ in topic_words_with_weights[:top_n] if isinstance(word, str)]
    
    if not topic_words:
        return 0

    umass_scores = []
    for i, word1 in enumerate(topic_words):
        for j, word2 in enumerate(topic_words):
            if i < j:
                try:
                    co_occurrence = dictionary.dfs[dictionary.token2id[word1]] + 1
                    word1_occurrence = dictionary.dfs[dictionary.token2id[word2]] + 1
                    umass = np.log((co_occurrence / word1_occurrence) + 1e-10)
                    umass_scores.append(umass)
                except KeyError:
                    continue

    return np.mean(umass_scores) if umass_scores else 0

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def calculate_coherence(model, data, tokenizer, bert_model, model_type, vectorizer, num_topics, topics):
    coherence_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for topic_words in topics:
        sentences = ["The topic is about " + word for word, _ in topic_words]
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        sentence_embeddings = outputs.last_hidden_state[:, 0, :]

        pairs = list(combinations(range(len(topic_words)), 2))

        if not pairs:
            coherence_scores.append(0)
            continue

        embeddings1 = sentence_embeddings[[i for i, j in pairs]]
        embeddings2 = sentence_embeddings[[j for i, j in pairs]]
        cosine_similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)

        coherence = cosine_similarities.mean().item()
        coherence_scores.append(coherence)

    return np.mean(coherence_scores) if coherence_scores else 0

def process_dataset(domain, dataset_name, data, model_types, num_topics_list, metrics_list):
    for model_type in model_types:
        for num_topics in num_topics_list:
            logging.info(f"Processing {domain} - {dataset_name} - {model_type} - {num_topics} topics")
            try:
                start_time = time.time()
                model, vectorizer, topics = perform_topic_modeling(data, num_topics, model_type)
                topic_modeling_time = time.time() - start_time

                start_time = time.time()
                coherence = calculate_coherence(model, data, tokenizer, bert_model, model_type, vectorizer, num_topics, topics)
                coherence_time = time.time() - start_time

                start_time = time.time()
                npmi = calculate_npmi(topics[0], data)
                npmi_time = time.time() - start_time

                start_time = time.time()
                tokenized_data = [simple_preprocess(doc) for doc in data]
                dictionary = Dictionary(tokenized_data)
                corpus = [dictionary.doc2bow(text) for text in tokenized_data]
                umass = calculate_umass(topics[0], corpus, dictionary)
                umass_time = time.time() - start_time

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
                logging.info(f"Computation times - Topic Modeling: {topic_modeling_time:.2f}s, Coherence: {coherence_time:.2f}s, NPMI: {npmi_time:.2f}s, U_Mass: {umass_time:.2f}s")

            except Exception as e:
                logging.error(f"Error processing {domain} - {dataset_name} - {model_type} - {num_topics}: {str(e)}")
                continue

    return metrics_list

if __name__ == '__main__':
    datasets = load_all_datasets()
    metrics_list = []
    computation_times = {}
    model_types = ['LDA', 'BERTopic', 'VAE']
    num_topics_list = [2, 4, 6, 8, 10]

    for domain, domain_datasets in datasets.items():
        for dataset_name, data in domain_datasets.items():
            metrics_list = process_dataset(domain, dataset_name, data, model_types, num_topics_list, metrics_list)

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
    
    # 모델별 평균 성능
    logging.info("\n모델별 평균 성능:")
    logging.info(metrics_df.groupby('Model')[['Coherence', 'NPMI', 'U_Mass']].mean())

    # 토픽 수별 평균 성능
    logging.info("\n토픽 수별 평균 성능:")
    logging.info(metrics_df.groupby('Num_Topics')[['Coherence', 'NPMI', 'U_Mass']].mean())

    # 도메인별 평균 성능
    logging.info("\n도메인별 평균 성능:")
    logging.info(metrics_df.groupby('Domain')[['Coherence', 'NPMI', 'U_Mass']].mean())

    # 시각화를 위한 라이브러리 import
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

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

    # 최고 성능 모델 식별
    best_models = {
        'Coherence': metrics_df.loc[metrics_df['Coherence'].idxmax()],
        'NPMI': metrics_df.loc[metrics_df['NPMI'].idxmax()],
        'U_Mass': metrics_df.loc[metrics_df['U_Mass'].idxmax()]
    }

    logging.info("\n최고 성능 모델:")
    for metric, best_model in best_models.items():
        logging.info(f"{metric}: {best_model['Model']} (토픽 수: {best_model['Num_Topics']}, 점수: {best_model[metric]:.4f})")

    # 결론 및 해석
    logging.info("\n결론 및 해석:")
    logging.info("1. 전반적으로 가장 좋은 성능을 보인 모델은 ...")
    logging.info("2. 토픽 수에 따른 성능 변화를 보면 ...")
    logging.info("3. 도메인별 성능 차이는 ...")
    logging.info("4. Coherence, NPMI, U_Mass 지표 간의 관계는 ...")
    logging.info("5. 추가적인 분석이 필요한 부분: ...")

    logging.info("\n분석이 완료되었습니다. 결과를 확인하세요.")
