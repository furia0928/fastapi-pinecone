from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# 📌 환경 변수에서 API 키 로드
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
INDEX_NAME = "markdown-index"

# 📌 Pinecone 클라이언트 초기화 (최신 방식)
pc = Pinecone(api_key=PINECONE_API_KEY)

# 📌 인덱스 가져오기
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    raise ValueError(f"Pinecone 인덱스 '{INDEX_NAME}'가 존재하지 않습니다. 먼저 생성하세요.")

index = pc.Index(INDEX_NAME)

# 📌 FastAPI 앱 생성
app = FastAPI()

# 📌 요청 모델 정의
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5  # 반환할 최대 개수

# 📌 OpenAI 임베딩 함수 (LangChain 사용)
def get_embedding(text: str):
    print(f"📌 임베딩 변환 시작: {text[:50]}...")  # 디버깅 출력

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=OPENAI_API_KEY,
    )

    response = embedding_model.embed_query(text)

    print(f"✅ 임베딩 변환 완료!")  # 완료 로그
    return response


@app.post("/search")
def search_vectors(request: SearchRequest):
    print(f"📌 검색 요청: {request.query}")  # 검색 요청 로그

    try:
        query_embedding = get_embedding(request.query)

        if query_embedding is None:
            raise HTTPException(status_code=400, detail="🚨 임베딩 변환 실패")

        # 📌 Pinecone 검색 수행
        print(f"📌 Pinecone 검색 시작 (namespace: langchain, top_k: {request.top_k})")

        results = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            namespace="langchain",  # 저장된 네임스페이스와 일치
            include_metadata=True
        )

        print(f"✅ 검색 완료! 결과 개수: {len(results.get('matches', []))}")
        print(f"📌 검색 결과: {results}")  # 전체 검색 결과 출력

        # 🚀 **예외 방지 코드 추가**
        if not results or "matches" not in results:
            print("🚨 Pinecone 검색 결과가 없음! 빈 리스트 반환")
            return {"matches": []}  # 빈 결과 반환

        # **🚨 중요: FastAPI가 JSON으로 변환할 수 있도록 데이터 정리**
        response = {"matches": []}
        for match in results["matches"]:
            response["matches"].append({
                "id": match.get("id", ""),
                "score": match.get("score", 0),
                "metadata": match.get("metadata", {})
            })

        return response  # ✅ FastAPI가 처리할 수 있도록 변환된 데이터 반환

    except Exception as e:
        print(f"🚨 FastAPI 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))



