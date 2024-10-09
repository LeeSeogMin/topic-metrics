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

        num_epochs = 5
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

def calculate_npmi(topic_words_with_weights, corpus, dictionary, top_n=10):
    """
    주어진 토픽의 NPMI 점수를 계산합니다.
    
    :param topic_words_with_weights: (list) 토픽 단어와 그에 대한 가중치 리스트
    :param corpus: (list) 문서 집합
    :param dictionary: (gensim Dictionary) 단어와 id의 매핑을 포함한 사전
    :param top_n: (int) 계산에 사용할 상위 N개의 토픽 단어 수
    :return: (float) NPMI 점수의 평균 값
    """
    topic_words = [word for word, _ in topic_words_with_weights[:top_n] if isinstance(word, str)]
    
    if not topic_words:
        logging.warning("토픽 단어가 없습니다.")
        return float('nan')  # 단어가 없는 경우 NaN 반환
    
    npmi_scores = []
    doc_count = len(corpus)  # 전체 문서 수

    # 두 단어 간의 NPMI 계산
    for i, word1 in enumerate(topic_words):
        for j, word2 in enumerate(topic_words):
            if i < j:  # 같은 단어 쌍은 계산하지 않음
                try:
                    word1_id = dictionary.token2id[word1]
                    word2_id = dictionary.token2id[word2]

                    # 문서에서 단어의 출현 확률 계산
                    doc_count_word1 = dictionary.dfs[word1_id]
                    doc_count_word2 = dictionary.dfs[word2_id]
                    co_doc_count = sum(1 for doc in corpus if word1_id in doc and word2_id in doc)

                    # PMI 및 NPMI 계산
                    p_w1 = doc_count_word1 / doc_count
                    p_w2 = doc_count_word2 / doc_count
                    p_w1_w2 = co_doc_count / doc_count

                    pmi = np.log(p_w1_w2 / (p_w1 * p_w2) + 1e-10)
                    npmi = pmi / (-np.log(p_w1_w2 + 1e-10))
                    npmi_scores.append(npmi)
                except KeyError:
                    logging.warning(f"단어 '{word1}' 또는 '{word2}'이 사전에 없습니다.")
                    continue

    # NPMI 점수 평균 반환 (유효한 점수가 없는 경우 NaN 반환)
    if not npmi_scores:
        logging.warning("NPMI 계산 중 유효한 쌍이 없음.")
        return float('nan')  # 빈 경우 NaN 반환

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

def calculate_coherence(model, data, tokenizer, bert_model, model_type, vectorizer, num_topics, topics):
    coherence_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for topic_words in topics:
        sentences = [f"The topic focuses on {word}" for word, _ in topic_words]
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
                tokenized_data = [simple_preprocess(doc) for doc in data]
                dictionary = Dictionary(tokenized_data)
                corpus = [dictionary.doc2bow(text) for text in tokenized_data]
                npmi = calculate_npmi(topics[0], corpus, dictionary)
                npmi_time = time.time() - start_time

                start_time = time.time()
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

# LLM 평가 관련 함수
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def call_openai_api(prompt: str, max_tokens: int = 3000) -> str:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    full_response = ""
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
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
일관성(Coherence) 평가 지표에 대해 설명드리겠습니다:

1. 일관성은 토픽 모델링의 품질을 평가하는 중요한 지표입니다.
2. 이 지표는 각 토픽 내의 단어들이 얼마나 의미적으로 연관되어 있는지를 측정합니다.
3. 일관성 점수는 주로 다음과 같은 방식으로 계산됩니다:
   a) 토픽 내 단어 쌍의 동시 출현 확률을 계산합니다.
   b) 이 확률들의 평균이나 합계를 구합니다.
4. 높은 일관성 점수는 토픽 내 단어들이 서로 밀접하게 관련되어 있음을 의미합니다.
5. 낮은 일관성 점수는 토픽이 덜 의미 있거나 일관성이 떨어짐을 나타냅니다.
6. 이 평가에서는 NPMI(Normalized Pointwise Mutual Information)와 C_V 일관성 지표를 사용합니다.

다음 토픽들을 평가해주세요. 각 토픽에 대해 1-10 척도로 일관성 점수를 매기고, 그 이유를 설명해주세요:

{topics}

평가 시 다음 사항을 고려해주세요:
1. 토픽 내 단어들이 얼마나 의미적으로 연관되어 있는지
2. 토픽이 얼마나 명확하고 해석 가능한지
3. 토픽 내 단어들이 일관된 주제나 개념을 나타내는지

각 토픽에 대해 다음 형식으로 응답해주세요:
토픽 X: [점수]
이유: [설명]
"""

    try:
        evaluation = call_openai_api(prompt)

        # 점수와 피드백 추출 로직
        topic_evaluations = re.findall(r"토픽 \d+:.*?(?=토픽 \d+:|$)", evaluation, re.DOTALL)
        for eval in topic_evaluations:
            match = re.search(r'(\d+)', eval)
            if match:
                topic_score = int(match.group(1))
                scores.append(topic_score)
                feedbacks.append(eval.strip())
            else:
                print(f"점수를 추출할 수 없습니다: {eval}")

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    return scores, feedbacks

def run_llm_evaluation(metrics_df, datasets, sample_size=100, chunk_size=10):
    llm_results = []
    actual_sample_size = min(sample_size, len(metrics_df))
    
    for index, row in tqdm(metrics_df.sample(n=actual_sample_size, random_state=42).iterrows(), total=actual_sample_size):
        domain = row['Domain']
        dataset_name = row['Dataset']
        model_type = row['Model']
        num_topics = row['Num_Topics']
        
        logging.info(f"LLM 평가 진행 중 - 도메인: {domain}, 데이터셋: {dataset_name}, 모델: {model_type}, 토픽 수: {num_topics}")

        try:
            data = datasets[domain][dataset_name]
            model, _, topics = perform_topic_modeling(data, num_topics, model_type)
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

            if len(llm_results) % chunk_size == 0:
                save_results_chunk(llm_results[-chunk_size:])
                
        except Exception as e:
            logging.error(f"Error processing {domain} - {dataset_name} - {model_type} - {num_topics}: {str(e)}")
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

def compare_metrics_with_llm(metrics_df, llm_df):
    merged_df = pd.merge(metrics_df, llm_df, on=['Domain', 'Dataset', 'Model', 'Num_Topics'])
    
    metrics = ['Coherence', 'NPMI', 'U_Mass']
    results = []

    for metric in metrics:
        pearson_corr, p_value = stats.pearsonr(merged_df[metric], merged_df['LLM_Avg_Score'])
        spearman_corr, p_value_spearman = stats.spearmanr(merged_df[metric], merged_df['LLM_Avg_Score'])
        mae = np.mean(np.abs(merged_df[metric] - merged_df['LLM_Avg_Score']))
        mse = np.mean((merged_df[metric] - merged_df['LLM_Avg_Score'])**2)
        
        results.append({
            'Metric': metric,
            'Pearson_Correlation': pearson_corr,
            'Spearman_Correlation': spearman_corr,
            'MAE': mae,
            'MSE': mse
        })

        # 산점도 그리기
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=metric, y='LLM_Avg_Score', data=merged_df)
        plt.title(f'{metric} vs LLM Score')
        plt.savefig(f'{metric}_vs_llm_scatter.png')
        plt.close()

        # Bland-Altman 플롯
        mean = (merged_df[metric] + merged_df['LLM_Avg_Score']) / 2
        diff = merged_df[metric] - merged_df['LLM_Avg_Score']
        md = np.mean(diff)
        sd = np.std(diff, axis=0)

        plt.figure(figsize=(10, 6))
        plt.scatter(mean, diff)
        plt.axhline(md, color='gray', linestyle='--')
        plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
        plt.title(f'Bland-Altman Plot: {metric} vs LLM Score')
        plt.xlabel('Mean of ' + metric + ' and LLM Score')
        plt.ylabel('Difference between ' + metric + ' and LLM Score')
        plt.savefig(f'{metric}_vs_llm_bland_altman.png')
        plt.close()

    results_df = pd.DataFrame(results)
    print("\n지표와 LLM 점수 비교 결과:")
    print(results_df)
    results_df.to_csv('metric_llm_comparison.csv', index=False)

    return results_df

def analyze_results_by_group(metrics_df, llm_df, group_by):
    merged_df = pd.merge(metrics_df, llm_df, on=['Domain', 'Dataset', 'Model', 'Num_Topics'])
    metrics = ['Coherence', 'NPMI', 'U_Mass']
    
    for group in merged_df[group_by].unique():
        group_df = merged_df[merged_df[group_by] == group]
        print(f"\n{group_by}: {group}")
        
        for metric in metrics:
            pearson_corr, _ = stats.pearsonr(group_df[metric], group_df['LLM_Avg_Score'])
            spearman_corr, _ = stats.spearmanr(group_df[metric], group_df['LLM_Avg_Score'])
            mae = np.mean(np.abs(group_df[metric] - group_df['LLM_Avg_Score']))
            mse = np.mean((group_df[metric] - group_df['LLM_Avg_Score'])**2)
            
            print(f"{metric}:")
            print(f"  Pearson Correlation: {pearson_corr:.4f}")
            print(f"  Spearman Correlation: {spearman_corr:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")

def summarize_results(metrics_df, llm_df):
    print("\n=== 결과 요약 ===")

    for groupby_col in ['Model', 'Num_Topics', 'Domain']:
        print(f"\n{groupby_col}별 평균 성능:")
        print(metrics_df.groupby(groupby_col)[['Coherence', 'NPMI', 'U_Mass']].mean())

    if llm_df is not None:
        print("\nLLM 평가 결과:")
        print(llm_df.groupby('Model')['LLM_Avg_Score'].mean())

    best_models = {
        'Coherence': metrics_df.loc[metrics_df['Coherence'].idxmax()],
        'NPMI': metrics_df.loc[metrics_df['NPMI'].idxmax()],
        'U_Mass': metrics_df.loc[metrics_df['U_Mass'].idxmax()]
    }

    print("\n최고 성능 모델:")
    for metric, best_model in best_models.items():
        print(f"{metric}: {best_model['Model']} (토픽 수: {best_model['Num_Topics']}, 점수: {best_model[metric]:.4f})")

    print("\n결론 및 해석:")
    print("1. 전반적으로 가장 좋은 성능을 보인 모델은 ...")
    print("2. 토픽 수에 따른 성능 변화를 보면 ...")
    print("3. 도메인별 성능 차이는 ...")
    print("4. Coherence, NPMI, U_Mass 지표 간의 관계는 ...")
    print("5. LLM 평가 결과와 자동 평가 지표 간의 일치도는 ...")

if __name__ == '__main__':
    datasets = load_all_datasets()
    metrics_list = []
    computation_times = {}
    model_types = ['LDA', 'BERTopic', 'VAE']
    num_topics_list = [2, 4]

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

    # LLM 평가 실행
    llm_df = run_llm_evaluation(metrics_df, datasets)

    # 지표와 LLM 점수 비교
    comparison_results = compare_metrics_with_llm(metrics_df, llm_df)

    # 모델별, 토픽 수별, 도메인별 분석
    for group_by in ['Model', 'Num_Topics', 'Domain']:
        analyze_results_by_group(metrics_df, llm_df, group_by)

    # 결과 해석
    print("\n결과 해석:")
    new_coherence_corr = comparison_results.loc[comparison_results['Metric'] == 'Coherence', 'Pearson_Correlation'].values[0]
    npmi_corr = comparison_results.loc[comparison_results['Metric'] == 'NPMI', 'Pearson_Correlation'].values[0]
    umass_corr = comparison_results.loc[comparison_results['Metric'] == 'U_Mass', 'Pearson_Correlation'].values[0]

    if new_coherence_corr > max(npmi_corr, umass_corr):
        print("1. 새로 개발된 Coherence 지표가 NPMI와 U_Mass보다 LLM 평가 결과와 더 높은 상관관계를 보입니다.")
        print("   이는 새로운 Coherence 지표가 전문가 평가를 더 잘 반영할 수 있음을 시사합니다.")
    else:
        print("1. 새로 개발된 Coherence 지표가 NPMI 또는 U_Mass보다 LLM 평가 결과와 낮은 상관관계를 보입니다.")
        print("   이는 새로운 Coherence 지표가 개선이 필요할 수 있음을 시사합니다.")

    print("\n2. 각 지표별 LLM 평가와의 상관관계:")
    print(f"   - 새로운 Coherence: {new_coherence_corr:.4f}")
    print(f"   - NPMI: {npmi_corr:.4f}")
    print(f"   - U_Mass: {umass_corr:.4f}")

    print("\n3. 결론:")
    if new_coherence_corr > max(npmi_corr, umass_corr):
        print("   개발된 Coherence 지표는 기존 지표들보다 LLM 평가 결과와 더 밀접한 관련을 보입니다.")
        print("   이는 새로운 지표가 토픽 모델의 품질을 평가하는 데 있어 더 타당성 있는 방법일 수 있음을 시사합니다.")
    else:
        print("   개발된 Coherence 지표는 기존 지표들보다 LLM 평가 결과와 덜 밀접한 관련을 보입니다.")
        print("   이는 새로운 지표가 추가적인 개선이 필요하거나, 특정 상황에서만 유효할 수 있음을 시사합니다.")

    print("\n4. 추가 고려사항:")
    print("   - 모델별, 토픽 수별, 도메인별 분석 결과를 검토하여 지표의 성능이 특정 조건에서 변화하는지 확인하세요.")
    print("   - Bland-Altman 플롯을 통해 지표와 LLM 평가 간의 일치도를 시각적으로 평가하세요.")
    print("   - MAE와 MSE를 비교하여 각 지표의 절대적인 오차를 고려하세요.")

    # 최종 결과 요약
    summarize_results(metrics_df, llm_df)

    logging.info("\n분석이 완료되었습니다. 결과를 확인하세요.")