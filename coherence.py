# 엉망 통합본
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim import models, corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from gensim.models.coherencemodel import CoherenceModel
import time
import json
from nltk.corpus import stopwords
from math import log
from itertools import combinations
from tqdm import tqdm
import logging
from collections import Counter, defaultdict
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
from sklearn.model_selection import train_test_split

# NLTK 데이터 다운로드 (한 번만 실행)
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# stop_words 정의
stop_words = set(stopwords.words('english'))

def load_data(file_path, sample_size=80):
    try:
        df = pd.read_csv(file_path, header=None, names=['text'])
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return []
    texts = df['text'].astype(str)
    if len(texts) > sample_size:
        texts = texts.sample(n=sample_size, random_state=42)
    print(f"Loaded {len(texts)} texts from {file_path}")
    return texts.tolist()

def load_all_datasets():
    datasets = {
        'academy': {
            'business': load_data('data/academy/business.csv')
        },
        'media': {
            'clothing_review': load_data('data/media/clothing_review.csv')
        },
        'news': {
            'agnews': load_data('data/news/agnews.csv')
        }
    }
    return datasets

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, latent_dim=None):
        if latent_dim is None:
            raise ValueError("latent_dim must be specified")
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
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def extract_vae_topics(vae_model, vectorizer, num_topics, top_n=10):
    with torch.no_grad():
        latent_vectors = torch.eye(num_topics).to(vae_model.fc3.weight.device)
        decoder_output = vae_model.decode(latent_vectors)
        decoder_output = decoder_output.cpu().numpy()

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_distribution in decoder_output:
        top_indices = topic_distribution.argsort()[-top_n:][::-1]
        topic_words = [feature_names[i] for i in top_indices]
        topics.append(topic_words)
    return topics

def perform_vae_topic_modeling(data, num_topics, num_epochs=5, hidden_dim=50):
    data = [str(doc) for doc in data if isinstance(doc, str) and len(doc) > 0]

    # TfidfVectorizer 사용
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(data)

    # MinMaxScaler를 사용하여 0-1 사이로 정규화
    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(doc_term_matrix.toarray())

    vocab_size = len(vectorizer.get_feature_names_out())
    input_dim = vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=num_topics).to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

    batch_size = 64
    data_loader = DataLoader(normalized_matrix.astype(np.float32), batch_size=batch_size, shuffle=True)

    vae_model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae_model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        logging.info(f"에폭 {epoch+1}/{num_epochs}, 손실: {train_loss / len(data_loader.dataset):.4f}")

    topics = extract_vae_topics(vae_model, vectorizer, num_topics)
    return vae_model, topics

def perform_bertopic_modeling(data):
    bertopic_model = BERTopic(language="english", calculate_probabilities=True)
    topics, _ = bertopic_model.fit_transform(data)
    
    num_topics = len(bertopic_model.get_topics()) - 1  # -1은 아웃라이어 토픽을 제외하기 위함
    topic_words = []
    for i in range(num_topics):
        words, _ = bertopic_model.get_topic(i)
        topic_words.append([word for word, _ in words[:10]])  # 상위 10개 단어만 추출
    
    return bertopic_model, topic_words, num_topics

def calculate_coherence(topics, tokenizer, bert_model):
    """
    Calculate the coherence score for given topics using BERT embeddings.

    Args:
    topics (list): List of topic word lists.
    tokenizer (BertTokenizer): BERT tokenizer.
    bert_model (BertModel): Pre-trained BERT model.

    Returns:
    float: Average coherence score across all topics.
    """
    coherence_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model.to(device)
    bert_model.eval()

    for topic_words in topics:
        inputs = tokenizer(topic_words, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 임베딩 사용

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

def process_metrics(domain, model_type, topics, data, metrics_list, tokenizer, bert_model):
    tokenized_data = [simple_preprocess(doc) for doc in data]
    dictionary = Dictionary(tokenized_data)
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]

    coherence = calculate_coherence(topics, tokenizer, bert_model)
    npmi = calculate_npmi(topics, corpus, dictionary)
    umass = calculate_umass(topics, corpus, dictionary)

    metrics_list.append({
        'Domain': domain,
        'Model': model_type,
        'Coherence': coherence,
        'NPMI': npmi,
        'U_Mass': umass
    })

    logging.info(f"Coherence: {coherence:.4f}, NPMI: {npmi:.4f}, U_Mass: {umass:.4f}")

def calculate_npmi(topics, corpus, dictionary, top_n=10):
    # 토픽에서 사용된 모든 단어의 집합 생성
    topic_words_set = set()
    for topic in topics:
        topic_words_set.update(topic[:top_n])

    # 단어를 ID로 매핑
    word2id = {word: dictionary.token2id[word] for word in topic_words_set if word in dictionary.token2id}
    id2word = {id: word for word, id in word2id.items()}

    # 단어와 단어 쌍의 문서 빈도 계산
    total_docs = len(corpus)
    word_doc_freq = defaultdict(int)
    pair_doc_freq = defaultdict(int)

    for doc in corpus:
        doc_word_ids = set([id for id, _ in doc])
        topic_word_ids_in_doc = doc_word_ids.intersection(set(word2id.values()))

        for word_id in topic_word_ids_in_doc:
            word_doc_freq[word_id] += 1

        for word_id1, word_id2 in combinations(topic_word_ids_in_doc, 2):
            pair = tuple(sorted((word_id1, word_id2)))
            pair_doc_freq[pair] += 1

    # NPMI 계산
    npmi_scores = []
    for topic in topics:
        topic_word_ids = [word2id[word] for word in topic[:top_n] if word in word2id]
        if len(topic_word_ids) < 2:
            continue
        pair_npmi_scores = []
        for word_id1, word_id2 in combinations(topic_word_ids, 2):
            pair = tuple(sorted((word_id1, word_id2)))
            co_doc_count = pair_doc_freq.get(pair, 0)
            if co_doc_count == 0:
                continue
            p_w1_w2 = co_doc_count / total_docs
            p_w1 = word_doc_freq[word_id1] / total_docs
            p_w2 = word_doc_freq[word_id2] / total_docs

            pmi = np.log(p_w1_w2 / (p_w1 * p_w2) + 1e-12)
            npmi = pmi / (-np.log(p_w1_w2 + 1e-12))
            pair_npmi_scores.append(npmi)
        if pair_npmi_scores:
            npmi_scores.append(np.mean(pair_npmi_scores))

    return np.mean(npmi_scores) if npmi_scores else float('nan')

def calculate_umass(topics, corpus, dictionary, top_n=10):
    # 토픽에서 사용된 모든 단어의 집합 생성
    topic_words_set = set()
    for topic in topics:
        topic_words_set.update(topic[:top_n])

    # 단어를 ID로 매핑
    word2id = {word: dictionary.token2id[word] for word in topic_words_set if word in dictionary.token2id}

    # 단어와 단어 쌍의 빈도 계산
    word_counts = defaultdict(int)
    pair_counts = defaultdict(int)

    for doc in corpus:
        doc_word_ids = set([id for id, _ in doc])
        topic_word_ids_in_doc = doc_word_ids.intersection(set(word2id.values()))

        for word_id in topic_word_ids_in_doc:
            word_counts[word_id] += 1

        for word_id1, word_id2 in combinations(topic_word_ids_in_doc, 2):
            pair = tuple(sorted((word_id1, word_id2)))
            pair_counts[pair] += 1

    # U_Mass 계산
    umass_scores = []
    for topic in topics:
        topic_word_ids = [word2id[word] for word in topic[:top_n] if word in word2id]
        if len(topic_word_ids) < 2:
            continue
        pair_umass_scores = []
        for i, word_id1 in enumerate(topic_word_ids):
            for word_id2 in topic_word_ids[:i]:
                pair = tuple(sorted((word_id1, word_id2)))
                co_occurrence = pair_counts.get(pair, 0) + 1  # 스무딩을 위해 +1
                word2_count = word_counts[word_id2] + 1  # 스무딩을 위해 +1
                umass = np.log(co_occurrence / word2_count)
                pair_umass_scores.append(umass)
        if pair_umass_scores:
            umass_scores.append(np.mean(pair_umass_scores))

    return np.mean(umass_scores) if umass_scores else float('nan')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

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
def analyze_stability(datasets, model_types, n_runs=10, sample_ratio=0.8):
    stability_results = []
    
    for domain, domain_datasets in datasets.items():
        # 각 도메인에서 첫 번째 데이터셋만 사용
        data = next(iter(domain_datasets.values()))
        
        # BERTopic으로 초기 토픽 수 결정
        _, _, num_topics = perform_bertopic_modeling(data)
        
        for model_type in model_types:
            metric_values = {
                'Coherence': [],
                'NPMI': [],
                'U_Mass': []
            }
            
            for _ in range(n_runs):
                sampled_data = np.random.choice(data, size=int(len(data) * sample_ratio), replace=False)
                
                if model_type == 'BERTopic':
                    model, topics, _ = perform_bertopic_modeling(sampled_data)
                elif model_type == 'VAE':
                    model, topics = perform_vae_topic_modeling(sampled_data, num_topics)
                
                tokenized_data = [simple_preprocess(doc) for doc in sampled_data]
                dictionary = Dictionary(tokenized_data)
                corpus = [dictionary.doc2bow(text) for text in tokenized_data]
                
                coherence = calculate_coherence(topics, tokenizer, bert_model)
                npmi = calculate_npmi(topics, corpus, dictionary)
                umass = calculate_umass(topics, corpus, dictionary)
                
                metric_values['Coherence'].append(coherence)
                metric_values['NPMI'].append(npmi)
                metric_values['U_Mass'].append(umass)
                
            for metric, values in metric_values.items():
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else float('nan')
                stability_results.append({
                    'Domain': domain,
                    'Model': model_type,
                    'Metric': metric,
                    'CV': cv
                })

    return pd.DataFrame(stability_results)

# 개선된 토픽 품질 시각화 함수
def visualize_topic_quality(metrics_df):
    from sklearn.preprocessing import StandardScaler
    
    metrics = ['Coherence', 'NPMI', 'U_Mass']
    metrics_df = metrics_df.dropna(subset=metrics)
    
    scaler = StandardScaler()
    scaled_metrics = scaler.fit_transform(metrics_df[metrics])
    
    mds = MDS(n_components=2, random_state=42)
    mds_coords = mds.fit_transform(scaled_metrics)
    
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

# LLM 평가 관련 함수
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def call_openai_api(prompt: str, max_tokens: int = 3000) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    full_response = ""
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in topic modeling and text analysis. Your task is to evaluate the coherence of topics based on provided documents."
                    },
                    {
                        "role": "user",
                        "content": prompt + ("\n\nContinue from: " + full_response if full_response else "")
                    }
                ],
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0.1,
                presence_penalty=0.1,
            )
            chunk = response.choices[0].message['content']
            full_response += chunk

            if response.choices[0].finish_reason != "length":
                break

            prompt = "Continue the previous response:"
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Retrying...")
            time.sleep(60)
        except openai.error.AuthenticationError:
            print("Authentication error. Check your API key.")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    return full_response

def llm_evaluation(topics, documents, model="gpt-4"):
    scores = []
    feedbacks = []

    if not isinstance(documents, list):
        documents = list(documents)

    prompt = f"""
Evaluate the following topics based on their coherence. Coherence is an important metric for assessing the quality of topic modeling:

1. Coherence measures how semantically related the words within each topic are.
2. It is typically calculated by considering the co-occurrence probabilities of word pairs within the topic.
3. Higher coherence scores indicate that the words in a topic are closely related and form a meaningful theme.
4. Lower coherence scores suggest that the topic may be less meaningful or coherent.

Please evaluate the following topics. For each topic, provide a coherence score on a scale of 1-10 and explain your reasoning:

{topics}

When evaluating, consider:
1. How semantically related are the words within each topic?
2. How clear and interpretable is the topic?
3. Do the words in the topic represent a consistent theme or concept?

Please respond for each topic in the following format:
Topic X: [score]
Reason: [explanation]
"""

    try:
        evaluation = call_openai_api(prompt)

        # Updated parsing logic to extract structured responses
        topic_evaluations = re.findall(r"Topic \d+:.*?(?=Topic \d+:|$)", evaluation, re.DOTALL)
        for eval in topic_evaluations:
            score_match = re.search(r'Topic (\d+):\s*(\d+)', eval)
            reason_match = re.search(r'Reason:\s*(.*)', eval, re.DOTALL)
            if score_match and reason_match:
                topic_score = int(score_match.group(2))
                if 1 <= topic_score <= 10:
                    scores.append(topic_score)
                    feedbacks.append(reason_match.group(1).strip())
                else:
                    print(f"Invalid score (not between 1 and 10): {eval}")
            else:
                print(f"Could not extract score or reason: {eval}")

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    return scores, feedbacks

def run_llm_evaluation(metrics_df, datasets, sample_size=100, chunk_size=10):
    llm_results = []
    actual_sample_size = min(sample_size, len(metrics_df))
    
    for index, row in tqdm(metrics_df.sample(n=actual_sample_size, random_state=42).iterrows(), total=actual_sample_size):
        domain = row['Domain']
        model_type = row['Model']
        
        logging.info(f"LLM 평가 진행 중 - 도메인: {domain}, 모델: {model_type}")

        try:
            # 각 도메인에서 첫 번째 데이터셋만 사용
            data = next(iter(datasets[domain].values()))
            if model_type == 'BERTopic':
                model, topics, _ = perform_bertopic_modeling(data)
            elif model_type == 'VAE':
                model, topics = perform_vae_topic_modeling(data, num_topics)
            else:
                continue
            
            scores, feedbacks = llm_evaluation(topics, data)

            result = {
                'Domain': domain,
                'Model': model_type,
                'LLM_Scores': scores,
                'LLM_Feedbacks': feedbacks
            }
            llm_results.append(result)

            if len(llm_results) % chunk_size == 0:
                save_results_chunk(llm_results[-chunk_size:])
                
        except Exception as e:
            logging.error(f"Error processing {domain} - {model_type}: {str(e)}")
            continue

    if len(llm_results) % chunk_size != 0:
        save_results_chunk(llm_results[-(len(llm_results) % chunk_size):])

    llm_df = pd.DataFrame(llm_results)
    return llm_df

def save_results_chunk(results_chunk):
    with open('llm_evaluation_results.json', 'a') as f:
        for result in results_chunk:
            json.dump(result, f)
            f.write('\n')

def analyze_llm_results(llm_df):
    llm_df['LLM_Avg_Score'] = llm_df['LLM_Scores'].apply(lambda scores: np.mean([s for s in scores if s is not None]))
    llm_df['LLM_Std_Score'] = llm_df['LLM_Scores'].apply(lambda scores: np.std([s for s in scores if s is not None]))
    llm_df['LLM_Median_Score'] = llm_df['LLM_Scores'].apply(lambda scores: np.median([s for s in scores if s is not None]))

    print("\nLLM 평가 결과:")
    print(llm_df[['Domain', 'Model', 'LLM_Avg_Score', 'LLM_Std_Score', 'LLM_Median_Score']])

def llm_auto_metric_correlation(metrics_df, llm_df):
    merged_df = pd.merge(metrics_df, llm_df, on=['Domain', 'Model'])

    metric_names = ['Coherence', 'NPMI', 'U_Mass']
    for metric in metric_names:
        valid_idx = merged_df['LLM_Avg_Score'].notnull()
        pearson_corr, p_value_pearson = stats.pearsonr(merged_df.loc[valid_idx, metric], merged_df.loc[valid_idx, 'LLM_Avg_Score'])
        spearman_corr, p_value_spearman = stats.spearmanr(merged_df.loc[valid_idx, metric], merged_df.loc[valid_idx, 'LLM_Avg_Score'])
        print(f"\nLLM 평가 점수와 {metric}의 상관관계:")
        print(f"Pearson: 상관계수 = {pearson_corr:.4f}, p-value = {p_value_pearson:.4f}")
        print(f"Spearman: 상관계수 = {spearman_corr:.4f}, p-value = {p_value_spearman:.4f}")

def verify_llm_consistency(topics, documents, n_repeats=5):
    all_scores = []
    for _ in range(n_repeats):
        scores, _ = llm_evaluation(topics, documents)
        all_scores.append(scores)
    all_scores = np.array(all_scores)
    std_scores = np.std(all_scores, axis=0)
    avg_std = np.mean(std_scores)
    cv_scores = std_scores / np.mean(all_scores, axis=0)
    avg_cv = np.mean(cv_scores)
    print(f"\nLLM 평가의 평균 표준편차: {avg_std:.4f}")
    print(f"LLM 평가의 평균 변동계수(CV): {avg_cv:.4f}")

def analyze_llm_feedback(llm_df):
    all_words = []
    for feedbacks in llm_df['LLM_Feedbacks']:
        for feedback in feedbacks:
            words = feedback.lower().split()
            all_words.extend([word for word in words if word not in stop_words])

    word_freq = Counter(all_words)
    print("\n피드백에서 가장 자주 등장하는 키워드:")
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}")

    coherence_keywords = ['coherent', 'consistent', 'related', 'connected', 'meaningful']
    print("\n일관성 관련 키워드 빈도:")
    for keyword in coherence_keywords:
        print(f"{keyword}: {word_freq[keyword]}")

    positive_keywords = ['good', 'great', 'excellent', 'well', 'clear']
    negative_keywords = ['poor', 'bad', 'unclear', 'confusing', 'unrelated']
    
    positive_count = sum(word_freq[word] for word in positive_keywords)
    negative_count = sum(word_freq[word] for word in negative_keywords)
    
    print(f"\n긍정적 피드백 키워드 수: {positive_count}")
    print(f"부정적 피드백 키워드 수: {negative_count}")

    relationship_keywords = ['related', 'similar', 'overlapping', 'connected', 'distinct']
    print("\n토픽 간 관계 관련 키워드 빈도:")
    for keyword in relationship_keywords:
        print(f"{keyword}: {word_freq[keyword]}")

    quality_keywords = ['coherent', 'meaningful', 'interpretable', 'clear', 'specific']
    print("\n토픽 품질 관련 키워드 빈도:")
    for keyword in quality_keywords:
        print(f"{keyword}: {word_freq[keyword]}")

    scores = [score for scores in llm_df['LLM_Scores'] for score in scores if score is not None]
    print("\n일관성 점수 분포:")
    print(f"평균: {np.mean(scores):.2f}")
    print(f"중앙값: {np.median(scores):.2f}")
    print(f"표준편차: {np.std(scores):.2f}")
    print(f"최소값: {np.min(scores):.2f}")
    print(f"최대값: {np.max(scores):.2f}")

    print("\n모델별 평균 일관성 점수:")
    for model in llm_df['Model'].unique():
        model_scores = [score for scores, m in zip(llm_df['LLM_Scores'], llm_df['Model']) 
                        for score in scores if score is not None and m == model]
        print(f"{model}: {np.mean(model_scores):.2f}")

def visualize_llm_results(llm_df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y='LLM_Avg_Score', data=llm_df)
    plt.title('모델별 LLM 평가 점수 분포')
    plt.savefig('llm_model_score_distribution.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Model', y='LLM_Avg_Score', data=llm_df)
    plt.title('모델별 LLM 평가 점수')
    plt.legend()
    plt.savefig('llm_model_score.png')
    plt.close()

def determine_optimal_topics(data, model_type, domain, dataset_name):
    if model_type == 'VAE':
        topic_numbers = range(5, 51, 5)  # 5부터 50까지 5 간격으로 토픽 수 설정
        coherence_scores = []
        
        for num_topics in topic_numbers:
            vae_model, topics = perform_vae_topic_modeling(data, latent_dim=num_topics)
            
            # coherence score 계산
            dictionary = Dictionary(data)
            corpus = [dictionary.doc2bow(text) for text in data]
            cm = CoherenceModel(topics=topics, texts=data, dictionary=dictionary, coherence='c_v')
            coherence_scores.append(cm.get_coherence())
        
        optimal_topics = topic_numbers[np.argmax(coherence_scores)]
        
    elif model_type == 'BERTopic':
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        best_score = -np.inf
        best_params = {}
        
        for min_topic_size in [10, 20, 30, 40, 50]:
            for n_gram_range in [(1, 1), (1, 2), (1, 3)]:
                model = BERTopic(n_gram_range=n_gram_range, min_topic_size=min_topic_size)
                topics, _ = model.fit_transform(train_data)
                score = model.score(test_data)  # BERTopic의 내장 평가 메서드 사용
                
                if score > best_score:
                    best_score = score
                    best_params = {'min_topic_size': min_topic_size, 'n_gram_range': n_gram_range}
        
        optimal_topics = best_params
    
    print(f"Optimal topics for {model_type} - {domain} - {dataset_name}: {optimal_topics}")
    return optimal_topics

def evaluate_coherence_stability(models, domains, datasets, n_runs=5):
    stability_results = []

    for model in models:
        for domain, data in zip(domains, datasets):
            coherence_values = []

            for _ in range(n_runs):
                # 모델 학습 및 Coherence 계산
                if model == 'Model A':
                    _, topics, _ = perform_bertopic_modeling(data)
                elif model == 'Model B':
                    _, topics = perform_vae_topic_modeling(data, num_topics)

                coherence = calculate_coherence(topics, tokenizer, bert_model)
                coherence_values.append(coherence)

            # 변동계수 계산
            cv = np.std(coherence_values) / np.mean(coherence_values) if np.mean(coherence_values) != 0 else float('nan')
            stability_results.append({
                'Model': model,
                'Domain': domain,
                'CV': cv
            })

    return pd.DataFrame(stability_results)

# Example usage
models = ['Model A', 'Model B']
domains = ['Domain 1', 'Domain 2', 'Domain 3']
datasets = [dataset1, dataset2, dataset3]  # Replace with actual datasets

stability_df = evaluate_coherence_stability(models, domains, datasets, n_runs=5)
print(stability_df)

def main():
    datasets = load_all_datasets()
    all_metrics = []
    model_types = ['BERTopic', 'VAE']
    
    for domain, domain_datasets in datasets.items():
        data = next(iter(domain_datasets.values()))
        
        bertopic_model, bertopic_topics, num_topics = perform_bertopic_modeling(data)  # Get num_topics
        
        for model_type in model_types:
            if model_type == 'BERTopic':
                metrics = process_metrics(domain, model_type, bertopic_topics, data, all_metrics, tokenizer, bert_model)
                all_metrics.extend(metrics)
            elif model_type == 'VAE':
                vae_model, vae_topics = perform_vae_topic_modeling(data, num_topics)  # Use num_topics
                metrics = process_metrics(domain, model_type, vae_topics, data, all_metrics, tokenizer, bert_model)
                all_metrics.extend(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('topic_modeling_metrics.csv', index=False)
    logging.info("Metrics saved to topic_modeling_metrics.csv")

    agreement_results = analyze_agreement(metrics_df)
    stability_df = analyze_stability(datasets, model_types)
    visualize_topic_quality(metrics_df)

    llm_df = run_llm_evaluation(metrics_df, datasets)
    
    analyze_llm_results(llm_df)
    llm_auto_metric_correlation(metrics_df, llm_df)
    
    sample_domain = list(datasets.keys())[0]
    sample_data = next(iter(datasets[sample_domain].values()))
    sample_model, sample_topics, _ = perform_bertopic_modeling(sample_data)
    verify_llm_consistency(sample_topics, sample_data)
    
    analyze_llm_feedback(llm_df)
    visualize_llm_results(llm_df)

    print_results(metrics_df, agreement_results, stability_df)

def print_results(metrics_df, agreement_results, stability_df):
    logging.info("\n=== 결과 분석 ===")
    
    logging.info("\n모델별 평균 성능:")
    logging.info(metrics_df.groupby('Model')[['Coherence', 'NPMI', 'U_Mass']].mean())

    logging.info("\n도메인별 평균 성능:")
    logging.info(metrics_df.groupby('Domain')[['Coherence', 'NPMI', 'U_Mass']].mean())

    logging.info("\n일치도 분석 결과 (Spearman 상관계수):")
    for pair, corr in agreement_results.items():
        logging.info(f"{pair}: {corr:.4f}")

    logging.info("\n안정성 분석 결과:")
    logging.info(stability_df.groupby(['Model', 'Metric'])['CV'].mean())

    logging.info("\n최고 성능 모델:")
    best_models = metrics_df.loc[metrics_df.groupby(['Domain'])['Coherence'].idxmax()]
    logging.info(best_models[['Domain', 'Model', 'Coherence']])

    logging.info("\n분석 완료. 결과를 확인하고 해석하세요.")

if __name__ == '__main__':
    main()