import os
import time
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
# 📌 환경 변수 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")

# 📌 Pinecone 클라이언트 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)

# 📌 인덱스 설정
INDEX_NAME = "markdown-index"
NAMESPACE = "langchain"

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

# 📌 Markdown 파일이 있는 폴더 경로
FOLDER_PATH = "./markdown_outputs"
BATCH_SIZE = 50  # 한 번에 업서트할 벡터 개수


# 📌 OpenAI 임베딩 생성 함수
def get_embedding(text):
    """텍스트를 OpenAI 임베딩으로 변환"""
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY
    )
    return embedding_model.embed_query(text)


# 📌 텍스트를 청크로 나누는 함수
def chunk_text(text, chunk_size=512, overlap=50):
    """텍스트를 일정 크기로 나누고, 겹치는 부분 유지"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks


# 📌 Markdown 파일을 Pinecone에 저장하는 함수
def process_markdown_files(folder_path):
    all_vectors = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # 📌 텍스트를 청크로 분할
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)

                vector_id = f"{filename}_{i}"
                metadata = {"filename": filename, "chunk_id": i, "text": chunk}

                all_vectors.append((vector_id, embedding, metadata))

                # 📌 BATCH 업서트
                if len(all_vectors) >= BATCH_SIZE:
                    index.upsert(vectors=all_vectors, namespace=NAMESPACE)
                    print(f"✅ {len(all_vectors)}개 벡터 저장 완료!")
                    all_vectors = []  # 리스트 초기화

    # 📌 마지막 남은 데이터 업서트
    if all_vectors:
        index.upsert(vectors=all_vectors, namespace=NAMESPACE)
        print(f"✅ 최종 {len(all_vectors)}개 벡터 저장 완료!")


# 📌 실행
process_markdown_files(FOLDER_PATH)

# 📌 벡터 저장 확인
time.sleep(5)  # Pinecone 반영 대기
stats = index.describe_index_stats()
print("📌 Pinecone 인덱스 상태:", stats)
print("🎯 모든 Markdown 파일이 Pinecone에 저장되었습니다!")
