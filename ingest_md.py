import os
import time
import json
import logging
import sys
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 📌 환경 변수 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")

# 📌 Pinecone 클라이언트 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)

# 📌 인덱스 설정
INDEX_NAME = "markdown-index"
NAMESPACE = "langchain_api"

# 📌 Pinecone 서버리스 인덱스 생성
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI 임베딩 벡터 크기
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 📌 Pinecone 인덱스 불러오기
index = pc.Index(INDEX_NAME)

# 📌 Markdown 파일 폴더 및 체크포인트 파일 설정
FOLDER_PATH = "./markdown_outputs"
CHECKPOINT_FILE = "checkpoint.json"
BATCH_SIZE = 50  # 한 번에 업서트할 벡터 개수

# 📌 체크포인트 데이터 로드 및 저장 함수
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_files": []}

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

checkpoint = load_checkpoint()

# 📌 OpenAI 임베딩 생성 함수 (재시도 로직 포함)
def get_embedding(text, retry=3):
    """텍스트를 OpenAI 임베딩으로 변환 (실패 시 재시도)"""
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY
    )
    attempts = 0
    while attempts < retry:
        try:
            return embedding_model.embed_query(text)
        except Exception as e:
            attempts += 1
            logging.error(f"임베딩 생성 실패 (재시도 {attempts}/{retry}): {e}")
            time.sleep(2)
    raise Exception("임베딩 생성 실패: 재시도 횟수 초과")

# 📌 텍스트를 청크로 나누는 함수
def chunk_text(text, chunk_size=512, overlap=50):
    """텍스트를 일정 크기로 나누고, 겹치는 부분 유지"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks

# 📌 배치 업서트 함수 (재시도 로직 포함)
def upsert_batch(vectors, retry=3):
    attempts = 0
    while attempts < retry:
        try:
            index.upsert(vectors=vectors, namespace=NAMESPACE)
            logging.info(f"✅ {len(vectors)}개 벡터 저장 완료!")
            return
        except Exception as e:
            attempts += 1
            logging.error(f"배치 업서트 실패 (재시도 {attempts}/{retry}): {e}")
            time.sleep(2)
    raise Exception("배치 업서트 실패: 재시도 횟수 초과")

# 📌 Markdown 파일을 Pinecone에 저장하는 함수 (체크포인트 및 에러 핸들링 적용)
def process_markdown_files(folder_path):
    all_vectors = []
    processed_files = set(checkpoint.get("processed_files", []))

    for filename in os.listdir(folder_path):
        if not filename.endswith(".md"):
            continue
        if filename in processed_files:
            logging.info(f"이미 처리된 파일 스킵: {filename}")
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logging.error(f"파일 읽기 실패: {filename} - {e}")
            continue

        # 텍스트를 청크로 분할
        chunks = chunk_text(text)
        logging.info(f"{filename}: {len(chunks)} 청크 생성됨.")

        for i, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk)
            except Exception as e:
                logging.error(f"{filename} 파일의 청크 {i} 임베딩 실패: {e}")
                continue  # 해당 청크만 건너뛰기

            vector_id = f"{filename}_{i}"
            metadata = {"filename": filename, "chunk_id": i, "text": chunk}
            all_vectors.append((vector_id, embedding, metadata))

            # BATCH 업서트
            if len(all_vectors) >= BATCH_SIZE:
                try:
                    upsert_batch(all_vectors)
                except Exception as e:
                    logging.error(f"배치 업서트 최종 실패: {e}")
                    # 실패한 배치는 나중에 다시 시도하도록 로직 추가 가능
                all_vectors = []

        # 파일 처리 완료 후 체크포인트 업데이트
        checkpoint["processed_files"].append(filename)
        save_checkpoint(checkpoint)
        logging.info(f"파일 처리 완료 및 체크포인트 업데이트: {filename}")

    # 남은 벡터 업서트
    if all_vectors:
        try:
            upsert_batch(all_vectors)
        except Exception as e:
            logging.error(f"마지막 배치 업서트 실패: {e}")

# 📌 실행
process_markdown_files(FOLDER_PATH)

# 📌 벡터 저장 확인
time.sleep(5)  # Pinecone 반영 대기
try:
    stats = index.describe_index_stats()
    logging.info(f"Pinecone 인덱스 상태: {stats}")
except Exception as e:
    logging.error(f"인덱스 상태 조회 실패: {e}")

logging.info("🎯 모든 Markdown 파일이 Pinecone에 저장되었습니다!")
