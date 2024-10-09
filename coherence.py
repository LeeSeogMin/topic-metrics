# -*- coding: utf-8 -*-
"""
코드 전체 구조
1. 필요한 라이브러리 import
2. 데이터 로드 및 전처리 함수들
3. 토픽 모델링 함수들
4. 평가 지표 계산 함수들 (coherence, NPMI, C_V)
5. LLM 평가 관련 함수들
6. 상관관계 분석 함수
7. summarize_results 함수
8. 메인 실행 코드
"""

# 필요한 모듈 import
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gensim
from gensim import models, corpora, matutils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from gensim.models.coherencemodel import CoherenceModel
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import os
import re
import json
import time
from nltk.corpus import stopwords
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal
from tenacity import retry, stop_after_attempt, wait_random_exponential
from bertopic import BERTopic
from transformers import BertTokenizer, BertModel
from math import log
from itertools import combinations
from tqdm import tqdm
import logging
from collections import Counter
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# stop_words 정의
stop_words = set(stopwords.words('english'))

def load_data(file_path, sample_size=100):
    df = pd.read_csv(file_path, header=None, names=['text'])
    texts = df['text'].astype(str)
    if len(texts) > sample_size:
        texts = texts.sample(n=sample_size, random_state=42)
    print(f"Loaded {len(texts)} texts from {file_path}")
    return texts.tolist()

# 데이터셋 로드
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
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        # Decoder
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
        return torch.sigmoid(self.fc4(h3))  # Changed to sigmoid

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def perform_topic_modeling(data, num_topics, model_type):
    # Ensure all data is of type string
    data = [str(doc) for doc in data if isinstance(doc, str) or pd.notna(doc)]

    # Check if num_topics is greater than the number of documents
    if num_topics > len(data):
        print(f"Adjusting num_topics from {num_topics} to {len(data)}")
        num_topics = len(data)

    # Common vectorizer
    vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'LDA':
        corpus = matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)
        id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())
        lda_model = models.LdaMulticore(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            workers=2,
            passes=10,
            random_state=42
        )
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
        input_dim = doc_term_matrix.shape[1]
        hidden_dim = 256
        latent_dim = num_topics

        vae_model = VAE(input_dim, hidden_dim, latent_dim).to(device)
        optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

        num_epochs = 2
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

        latent_vectors = []
        with torch.no_grad():
            for i in range(0, doc_term_matrix.shape[0], 128):
                batch = torch.FloatTensor(doc_term_matrix[i:i+128].toarray()).to(device)
                mu, logvar = vae_model.encode(batch)
                z = vae_model.reparameterize(mu, logvar)
                latent_vectors.append(z.cpu().numpy())

        latent_vectors = np.vstack(latent_vectors)
        kmeans = KMeans(n_clusters=num_topics, random_state=42).fit(latent_vectors)

        topics = [[] for _ in range(num_topics)]
        for idx, label in enumerate(kmeans.labels_):
            doc = data[idx]
            topics[label].extend(doc.split())

        topics = [list(pd.Series(words).value_counts().index[:10]) for words in topics]

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

# 7. 평가 지표 계산 함수
def calculate_npmi(topic_words_with_weights, texts, top_n=10):
    # 단어들만 추출하고, 문자열인지 확인
    topic_words = [word for word, _ in topic_words_with_weights[:top_n] if isinstance(word, str)]
    
    if not topic_words:
        return 0  # 토픽 단어가 없으면 0 반환

    vectorizer = CountVectorizer(vocabulary=topic_words)
    doc_word_matrix = vectorizer.fit_transform(texts)
    
    word_doc_counts = doc_word_matrix.sum(axis=0).A1  # 각 단어의 문서 내 등장 횟수
    doc_count = len(texts)
    
    npmi_scores = []
    for i, word1 in enumerate(topic_words):
        for j, word2 in enumerate(topic_words):
            if i < j:
                idx1 = vectorizer.vocabulary_[word1]
                idx2 = vectorizer.vocabulary_[word2]
                co_doc_count = doc_word_matrix[:, idx1].multiply(doc_word_matrix[:, idx2]).nnz
                if co_doc_count == 0:
                    continue  # 공출현 빈도�� 0이면 건너뜁니다.

                p_w1 = word_doc_counts[idx1] / doc_count
                p_w2 = word_doc_counts[idx2] / doc_count
                p_w1_w2 = co_doc_count / doc_count

                pmi = np.log(p_w1_w2 / (p_w1 * p_w2) + 1e-10)
                npmi = pmi / (-np.log(p_w1_w2 + 1e-10))
                npmi_scores.append(npmi)

    if npmi_scores:
        return np.mean(npmi_scores)
    else:
        return 0  # 계산된 NPMI 점수가 없으면 0 반환

def calculate_umass(topic_words_with_weights, corpus, dictionary, top_n=10):
    topic_words = [word for word, _ in topic_words_with_weights[:top_n] if isinstance(word, str)]
    
    if not topic_words:
        return 0  # Return 0 if there are no topic words

    umass_scores = []
    for i, word1 in enumerate(topic_words):
        for j, word2 in enumerate(topic_words):
            if i < j:
                try:
                    # Calculate conditional probability
                    co_occurrence = dictionary.dfs[dictionary.token2id[word1]] + 1
                    word1_occurrence = dictionary.dfs[dictionary.token2id[word2]] + 1
                    umass = np.log((co_occurrence / word1_occurrence) + 1e-10)
                    umass_scores.append(umass)
                except KeyError:
                    continue  # Skip if the word is not in the dictionary

    if umass_scores:
        return np.mean(umass_scores)
    else:
        return 0  # Return 0 if no U_Mass scores were calculated

# BERT 델과 토크나저 초기
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def calculate_coherence(model, data, tokenizer, bert_model, model_type, vectorizer, num_topics, topics):
    coherence_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 타입에 따라 토픽 추출 방식 변경
    if model_type == 'LDA':
        topics = [model.show_topic(i, topn=10) for i in range(num_topics)]
    elif model_type == 'BERTopic':
        topics = [model.get_topic(i) for i in range(num_topics) if model.get_topic(i)]
    elif model_type == 'VAE':
        # VAE의 경우 이미 topics가 제공됨
        pass
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    for topic_words in topics:
        # 토픽 단어들을 문장 형태로 변환하여 문맥을 고려한 임베딩 계산
        sentences = ["The topic is about " + word for word, _ in topic_words]
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        # 문장 임베딩은 last_hidden_state의 [CLS] 토큰 벡터 사용
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # 단어 쌍의 조합 생성 (중복 없이)
        pairs = list(combinations(range(len(topic_words)), 2))

        if not pairs:
            coherence_scores.append(0)
            continue

        # 코사인 유사도 계산
        embeddings1 = sentence_embeddings[[i for i, j in pairs]]
        embeddings2 = sentence_embeddings[[j for i, j in pairs]]
        cosine_similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)

        # coherence 점수는 코사인 유사도의 평균값
        coherence = cosine_similarities.mean().item()
        coherence_scores.append(coherence)

    if coherence_scores:
        return np.mean(coherence_scores)
    else:
        return 0

def calculate_evaluation_metrics(model, data, model_type, vectorizer, num_topics, topics):
    # Load BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)

    # Coherence calculation
    coherence = calculate_coherence(model, data, tokenizer, bert_model, model_type, vectorizer, num_topics, topics)

    # NPMI calculation
    if topics:
        topic_words_with_weights = topics[0]  # Use the first topic
    else:
        topic_words_with_weights = []

    npmi = calculate_npmi(topic_words_with_weights, data)

    # U_Mass calculation
    # For U_Mass, we need tokenized data
    tokenized_data = [simple_preprocess(doc) for doc in data]
    dictionary = Dictionary(tokenized_data)
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]
    umass = calculate_umass(topic_words_with_weights, corpus, dictionary)

    return coherence, npmi, umass


# 8. LLM 평가 관련 함수
def call_openai_api(prompt: str, max_tokens: int = 3000) -> str:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    full_response = ""
    while True:
        try:
            response = client.chat.completions.create(
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
            chunk = response.choices[0].message.content
            full_response += chunk

            if not response.choices[0].finish_reason == "length":
                break

            prompt = "Continue the previous response:"
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Retrying...")
            raise
        except openai.error.AuthenticationError:
            print("Authentication error. Check your API key.")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    return full_response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def llm_evaluation(topics, documents, model="gpt-4o-mini"):
    scores = []
    feedbacks = []

    if not isinstance(documents, list):
        documents = list(documents)

    prompt = f"""
일관성(Coherence) 평가 지표 대해 설명드리겠습다:

1. 일관성은 토픽 모델링의 품질을 평가하는 중요한 지표입니다.
2. 이 지표는 각 토픽 내의 단어 얼마나 의미적으�� 연관되어 있는지를 측정합니다.
3. 일관성 점수는 주로 다과 같은 방식으로 계산됩니다:
   a) 토픽 내 단어 쌍의 동시 출현 확률을 계산합니다.
   b) 이 확률들의 평균이나 합계를 구합니.
4. 높은 일관성 점수는 토픽 내 단어들이 서로 밀접게 관련되어 있음을 의미합니다.
5. 낮은 일관성 점수는 토픽이 덜 의미 있거나 일관이 떨어짐을 나타냅다.
6. 이 평가에서는 NPMI(Normalized Pointwise Mutual Information)와 C_V 일관성 지표 사용합니다.

다음 토픽들을 평가해주세요. 각 토픽에 대해 1-10 척도로 일관성 점수를 매기, 그 이유를 설명해주세요:

{topics}

평가 시 다음 사항을 고려해주세요:
1. 토픽 내 단어들이 얼마나 의미적으로 연관되어 있는지
2. 토픽이 얼마나 명확하고 해석 가능한지
3. 토픽 내 단어들이 일관된 주제나 개념을 나타내는지

각 토픽에 대해 다음 형식으로 응답해주요:
토픽 X: [점수]
이유: [설명]
"""

    try:
        evaluation = call_openai_api(prompt)

        # 점수와 피백 추출 로직
        topic_evaluations = re.findall(r"토픽 \d+:.*?(?=토픽 \d+:|$)", evaluation, re.DOTALL)
        for eval in topic_evaluations:
            match = re.search(r'(\d+)', eval)
            if match:
                topic_score = int(match.group(1))
                scores.append(topic_score)
                feedbacks.append(eval.strip())
            else:
                print(f"점수를 추출할 수 없습니: {eval}")

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    return scores, feedbacks

def run_llm_evaluation(metrics_df, datasets, sample_size=100, chunk_size=10):
    llm_results = []
    actual_sample_size = min(sample_size, len(metrics_df))
    
    # 로 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 진행 상황을 표시하기 위한 tqdm 사용
    for index, row in tqdm(metrics_df.sample(n=actual_sample_size, random_state=42).iterrows(), total=actual_sample_size):
        domain = row['Domain']
        dataset_name = row['Dataset']
        model_type = row['Model']
        num_topics = row['Num_Topics']
        
        logging.info(f"LLM 평가 진행 중 - 도인: {domain}, 데이터셋: {dataset_name}, 모델: {model_type}, 토픽 수: {num_topics}")

        try:
            data = datasets[domain][dataset_name]

            # 토픽 모델링 수행 (이미 계산된 모델이 있다면 그것을 사용)
            model_key = f"{domain}_{dataset_name}_{model_type}_{num_topics}"
            if model_key in pre_computed_models:
                model, topics = pre_computed_models[model_key]
            else:
                if model_type == 'VAE':
                    model, _, topics = perform_topic_modeling(data, num_topics, model_type)
                else:
                    model, _ = perform_topic_modeling(data, num_topics, model_type)
                    topics = extract_topics(model_type, model, num_topics)
                
                # 계산된 모델 저장
                pre_computed_models[model_key] = (model, topics)

            # LLM 평가 수행
            scores, feedbacks = llm_evaluation(topics, data)

            result = {
                'Domain': domain,
                'Dataset': dataset_name,
                'Model': model_type,
                'Num_Topics': num_topics,
                'LLM_Scores': scores,
                'LLM_Feedbacks': feedbacks
            }
            llm_results.append(result)

            # 메모리 관리: 청크 단위로 결과 저장
            if len(llm_results) % chunk_size == 0:
                save_results_chunk(llm_results[-chunk_size:])
                
        except Exception as e:
            logging.error(f"Error processing {domain} - {dataset_name} - {model_type} - {num_topics}: {str(e)}")
            continue

    # 남은 결과 저장
    if len(llm_results) % chunk_size != 0:
        save_results_chunk(llm_results[-(len(llm_results) % chunk_size):])

    llm_df = pd.DataFrame(llm_results)
    return llm_df

def save_results_chunk(results_chunk):
    with open('llm_evaluation_results.json', 'a') as f:
        for result in results_chunk:
            json.dump(result, f)
            f.write('\n')

# 미리 계산된 모델을 저장할 딕셔너리
pre_computed_models = {}

def analyze_llm_results(llm_df):
    llm_df['LLM_Avg_Score'] = llm_df['LLM_Scores'].apply(lambda scores: np.mean([s for s in scores if s is not None]))
    llm_df['LLM_Std_Score'] = llm_df['LLM_Scores'].apply(lambda scores: np.std([s for s in scores if s is not None]))
    llm_df['LLM_Median_Score'] = llm_df['LLM_Scores'].apply(lambda scores: np.median([s for s in scores if s is not None]))

    print("\nLLM 평가 결과:")
    print(llm_df[['Domain', 'Model', 'Num_Topics', 'LLM_Avg_Score', 'LLM_Std_Score', 'LLM_Median_Score']])

def llm_auto_metric_correlation(metrics_df, llm_df):
    merged_df = pd.merge(metrics_df, llm_df, on=['Domain', 'Dataset', 'Model', 'Num_Topics'])

    metric_names = ['Coherence', 'NPMI', 'C_V']
    for metric in metric_names:
        valid_idx = merged_df['LLM_Avg_Score'].notnull()
        pearson_corr, p_value_pearson = pearsonr(merged_df.loc[valid_idx, metric], merged_df.loc[valid_idx, 'LLM_Avg_Score'])
        spearman_corr, p_value_spearman = spearmanr(merged_df.loc[valid_idx, metric], merged_df.loc[valid_idx, 'LLM_Avg_Score'])
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
    # 피드백에서 자주 등장하는 키워드 추출
    all_words = []
    for feedbacks in llm_df['LLM_Feedbacks']:
        for feedback in feedbacks:
            words = feedback.lower().split()
            all_words.extend([word for word in words if word not in stop_words])

    word_freq = Counter(all_words)
    print("\n피드백에서 가장 자주 등장하는 키워드:")
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}")

    # 일관성 관련 키워드 분석
    coherence_keywords = ['coherent', 'consistent', 'related', 'connected', 'meaningful']
    print("\n일관성 관련 키워드 빈도:")
    for keyword in coherence_keywords:
        print(f"{keyword}: {word_freq[keyword]}")

    # 긍정적/정적 피드백 분석
    positive_keywords = ['good', 'great', 'excellent', 'well', 'clear']
    negative_keywords = ['poor', 'bad', 'unclear', 'confusing', 'unrelated']
    
    positive_count = sum(word_freq[word] for word in positive_keywords)
    negative_count = sum(word_freq[word] for word in negative_keywords)
    
    print(f"\n긍정적 피드백 키워드 수: {positive_count}")
    print(f"부정적 피드백 키워드 수: {negative_count}")

    # 토픽 간 관계 분석
    relationship_keywords = ['related', 'similar', 'overlapping', 'connected', 'distinct']
    print("\n픽 간 관계 관련 키워드 빈도:")
    for keyword in relationship_keywords:
        print(f"{keyword}: {word_freq[keyword]}")

    # 토픽 품질 분석
    quality_keywords = ['coherent', 'meaningful', 'interpretable', 'clear', 'specific']
    print("\n토 품질 관련 키워드 빈도:")
    for keyword in quality_keywords:
        print(f"{keyword}: {word_freq[keyword]}")

    # 일관성 점수 분포 분석
    scores = [score for scores in llm_df['LLM_Scores'] for score in scores if score is not None]
    print("\n일관성 점수 분포:")
    print(f"평균: {np.mean(scores):.2f}")
    print(f"중앙값: {np.median(scores):.2f}")
    print(f"표준편차: {np.std(scores):.2f}")
    print(f"최소: {np.min(scores):.2f}")
    print(f"최대값: {np.max(scores):.2f}")

    # 모델별 일관 점수 비교
    print("\n모델별 평균 일관성 점수:")
    for model in llm_df['Model'].unique():
        model_scores = [score for scores, m in zip(llm_df['LLM_Scores'], llm_df['Model']) 
                        for score in scores if score is not None and m == model]
        print(f"{model}: {np.mean(model_scores):.2f}")

    # 토픽 수에 따른 일관성 점수 변화
    print("\n 수에 따른 평균 일관성 점수:")
    for num_topics in sorted(llm_df['Num_Topics'].unique()):
        topic_scores = [score for scores, n in zip(llm_df['LLM_Scores'], llm_df['Num_Topics']) 
                        for score in scores if score is not None and n == num_topics]
        print(f"{num_topics} 토픽: {np.mean(topic_scores):.2f}")

# 9. 결과 분석 및 시각화 함���
def visualize_llm_results(llm_df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y='LLM_Avg_Score', data=llm_df)
    plt.title('모델별 LLM 평가 점수 분포')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Num_Topics', y='LLM_Avg_Score', hue='Model', data=llm_df)
    plt.title('토픽 수에 따른 LLM 평가 점수')
    plt.show()

def summarize_results(metrics_df, llm_df):
    print("\n=== 결과 요약 ===")

    # 모델별, 토픽 수별, 도메별 평균 성능 비교
    for groupby_col in ['Model', 'Num_Topics', 'Domain']:
        print(f"\n{groupby_col}별 평균 성능:")
        print(metrics_df.groupby(groupby_col)[['Coherence', 'NPMI', 'C_V']].mean())

    # LLM 평가 결과
    if llm_df is not None:
        print("\nLLM 평가 결과:")
        print(llm_df.groupby('Model')['LLM_Avg_Score'].mean())

    # 최고 성능 모델
    best_models = {
        'Coherence': metrics_df.loc[metrics_df['Coherence'].idxmax()],
        'NPMI': metrics_df.loc[metrics_df['NPMI'].idxmax()],
        'C_V': metrics_df.loc[metrics_df['C_V'].idxmax()]
    }

    print("\n최고 성능 모델:")
    for metric, best_model in best_models.items():
        print(f"{metric}: {best_model['Model']} (토픽 수: {best_model['Num_Topics']}, 점수: {best_model[metric]:.4f})")

    # 결론 및 해석
    print("\n결론  해석:")
    print("1. 전반적으로 가장 좋은 성능을 보인 모델은 ...")
    print("2. 토픽 수에 따른 성능 변화를 보면 ...")
    print("3. 도메인별 성능 차이는 ...")
    print("4. Coherence, NPMI, C_V 지표 간의 계는 ...")
    print("5. LLM 평가 결과와 자동 평가 지표 간의 일치도는 ...")

def extract_topics(model_type, model, num_topics):
    if model_type == 'LDA':
        return [model.show_topic(i, topn=10) for i in range(num_topics)]
    elif model_type == 'BERTopic':
        return [model.get_topic(i) for i in range(num_topics) if model.get_topic(i)]
    elif model_type == 'VAE':
        # VAE의 경우 토픽 추출 로직을 여기에 구현
        pass
    else:
        raise ValueError("Invalid model type")

def correlation_analysis(metrics_df):
    # 분할 지표들
    metrics = ['Coherence', 'NPMI', 'C_V', 'LLM_Avg_Score']

    # 상관계수 계산
    corr_matrix = metrics_df[metrics].corr()

    # 히트맵 생성
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Coherence Metrics')
    plt.tight_layout()
    plt.savefig('coherence_correlation_heatmap.png')
    plt.close()

    print("\n상관계수 매트릭스:")
    print(corr_matrix)

    # 각 지표 쌍에 대한 상관관계 분석
    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            metric1 = metrics[i]
            metric2 = metrics[j]
            
            # Pearson 상관계수
            pearson_corr, p_value = stats.pearsonr(metrics_df[metric1], metrics_df[metric2])
            print(f"\n{metric1}와 {metric2}의 Pearson 상관계수: {pearson_corr:.4f}")
            print(f"p-value: {p_value:.4f}")

            # Spearman 순위 상관계수
            spearman_corr, p_value = stats.spearmanr(metrics_df[metric1], metrics_df[metric2])
            print(f"{metric1}와 {metric2}의 Spearman 순위 상관계수: {spearman_corr:.4f}")
            print(f"p-value: {p_value:.4f}")

            # 산점도 생성
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=metric1, y=metric2, data=metrics_df)
            plt.title(f'{metric1} vs {metric2}')
            plt.tight_layout()
            plt.savefig(f'coherence_{metric1}_vs_{metric2}_scatter.png')
            plt.close()

    # 모델 유형별 분석
    for model in metrics_df['Model'].unique():
        model_df = metrics_df[metrics_df['Model'] == model]
        print(f"\n모델 {model}에 대한 상관계수:")
        print(model_df[metrics].corr())

    # 토픽 수별 분석
    for num_topics in metrics_df['Num_Topics'].unique():
        topic_df = metrics_df[metrics_df['Num_Topics'] == num_topics]
        print(f"\n토픽 수 {num_topics}에 대한 상관계수:")
        print(topic_df[metrics].corr())
        
def process_dataset(domain, dataset_name, data, model_types, num_topics_list, metrics_list):
    for model_type in model_types:
        for num_topics in num_topics_list:
            print(f"\nProcessing {domain} - {dataset_name}")
            print(f"Model: {model_type}, Num Topics: {num_topics}")
            try:
                model, vectorizer, topics = perform_topic_modeling(data, num_topics, model_type)
                coherence = calculate_coherence(model, data, tokenizer, bert_model, model_type, vectorizer, num_topics, topics)
                npmi = calculate_npmi(topics[0], data)
                umass = calculate_umass(topics[0], corpus, dictionary)
                
                print(f"Coherence: {coherence:.4f}")
                print(f"NPMI: {npmi:.4f}")
                print(f"U_Mass: {umass:.4f}")
                
                # 결과를 metrics_list에 추가하는 코드
                metrics_list.append({
                    'Domain': domain,
                    'Dataset': dataset_name,
                    'Model': model_type,
                    'Num_Topics': num_topics,
                    'Coherence': coherence,
                    'NPMI': npmi,
                    'U_Mass': umass
                })
                
            except Exception as e:
                print(f"Error processing {domain} - {dataset_name} - {model_type} - {num_topics}: {str(e)}")
                continue

    return metrics_list        

# 10. 메인 실행 코드
if __name__ == '__main__':
    # 데이터를 한 번만 로드
    datasets = load_all_datasets()

    metrics_list = []
    computation_times = {}
    model_types = ['LDA', 'BERTopic', 'VAE']
    num_topics_list = [2, 4]

    for domain, domain_datasets in datasets.items():
        for dataset_name, data in domain_datasets.items():
            print(f"\nProcessing {domain} - {dataset_name}")
            for model_type in model_types:
                for num_topics in num_topics_list:
                    print(f"\nModel: {model_type}, Num Topics: {num_topics}")
                    try:
                        start_time = time.time()
                        model, vectorizer, topics = perform_topic_modeling(data, num_topics, model_type)
                        topic_modeling_time = time.time() - start_time

                        start_time = time.time()
                        coherence = calculate_coherence(model, data, tokenizer, bert_model, model_type, vectorizer, num_topics, topics)
                        coherence_time = time.time() - start_time

                        if model_type == 'LDA':
                            topic_words_with_weights = model.show_topic(0, topn=10)
                        elif model_type == 'BERTopic':
                            topic_words_with_weights = model.get_topic(0)
                        elif model_type == 'VAE':
                            topic_words_with_weights = [(word, 1.0) for word in topics[0]]
                        else:
                            topic_words_with_weights = []

                        start_time = time.time()
                        npmi = calculate_npmi(topic_words_with_weights, data)
                        npmi_time = time.time() - start_time

                        start_time = time.time()
                        tokenized_data = [simple_preprocess(doc) for doc in data]
                        dictionary = Dictionary(tokenized_data)
                        corpus = [dictionary.doc2bow(text) for text in tokenized_data]
                        umass = calculate_umass(topic_words_with_weights, corpus, dictionary)
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

                        computation_times[f"{domain}_{dataset_name}_{model_type}_{num_topics}"] = {
                            'Topic Modeling': topic_modeling_time,
                            'Coherence': coherence_time,
                            'NPMI': npmi_time,
                            'U_Mass': umass_time
                        }

                        print(f"Coherence: {coherence:.4f}")
                        print(f"NPMI: {npmi:.4f}")
                        print(f"U_Mass: {umass:.4f}")

                    except Exception as e:
                        print(f"Error processing {domain} - {dataset_name} - {model_type} - {num_topics}: {str(e)}")
                        continue

    # 결과를 DataFrame으로 변환하고 CSV로 저장
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('topic_modeling_metrics.csv', index=False)
    print("\nMetrics saved to topic_modeling_metrics.csv")

    # 계산 시간을 JSON 파일로 저장
    with open('computation_times.json', 'w') as f:
        json.dump(computation_times, f)
    print("Computation times saved to computation_times.json")

    # 결과 분석 및 시각화 (이 부분은 필요에 따라 추가하거나 수정할 수 있습니다)
    print("\n=== 결과 분석 ===")
    
    # 모델별 평균 성능
    print("\n모델별 평균 성능:")
    print(metrics_df.groupby('Model')[['Coherence', 'NPMI', 'U_Mass']].mean())

    # 토픽 수별 평균 성능
    print("\n토픽 수별 평균 성능:")
    print(metrics_df.groupby('Num_Topics')[['Coherence', 'NPMI', 'U_Mass']].mean())

    # 도메인별 평균 성능
    print("\n도메인별 평균 성능:")
    print(metrics_df.groupby('Domain')[['Coherence', 'NPMI', 'U_Mass']].mean())

    # 시각화 (matplotlib이나 seaborn을 사용하여 그래프를 그릴 수 있습니다)
    # 예: plt.figure(figsize=(12, 6))
    #     sns.boxplot(x='Model', y='Coherence', data=metrics_df)
    #     plt.title('모델별 Coherence 분포')
    #     plt.savefig('model_coherence_distribution.png')
    #     plt.close()