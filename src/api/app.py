from time import perf_counter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from config.config import get_config
from src.api.routes import router
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager


class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> bool:
        async with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)

            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > cutoff
            ]

            if len(self.requests[client_id]) >= self.requests_per_minute:
                return False

            self.requests[client_id].append(now)
            return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: startup and shutdown management"""
    # Startup
    logger.info("🚀 RAG-Lines API starting up...")
    config = get_config()

    try:
        # Lazy initialize RAG pipeline
        from src.api.routes import get_pipeline
        _ = get_pipeline()
        logger.info("✅ RAG Pipeline initialized")
    except Exception as e:
        logger.warning(f"⚠️  RAG Pipeline initialization deferred: {e}")

    try:
        # Test vector store
        from src.vectorstore.chroma_db import ChromaVectorStore
        vs = ChromaVectorStore()
        logger.info(f"✅ Vector store ready at {vs.db_path}")
    except Exception as e:
        logger.warning(f"⚠️  Vector store check: {e}")

    logger.info("✅ All startup checks complete")
    yield
    # Shutdown
    logger.info("🛑 Shutting down...")


def create_app() -> FastAPI:
    config = get_config()
    rate_limiter = RateLimiter(requests_per_minute=config.api_rate_limit_per_minute)

    app = FastAPI(
        title="RAG-Lines API",
        description="Retrieval-Augmented Generation System API",
        version="1.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"

        is_allowed = await rate_limiter.is_allowed(client_ip)
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Max 100 requests per minute."}
            )

        return await call_next(request)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = perf_counter()
        logger.info(f"HTTP {request.method} {request.url.path} started from {request.client.host if request.client else 'unknown'}")
        response = await call_next(request)
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info(
            f"HTTP {request.method} {request.url.path} completed status={response.status_code} elapsed_ms={elapsed_ms:.2f}"
        )
        return response

    app.include_router(router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "rag-lines-api", "version": "1.2.0"}

    @app.get("/health/live")
    async def health_live():
        """Liveness probe - is the app running?"""
        return {"status": "alive", "timestamp": datetime.now().isoformat()}

    @app.get("/health/ready")
    async def health_ready():
        """Readiness probe - are all dependencies available?"""
        checks = {}

        # Check LLM server
        try:
            from src.llm.local_llm import LocalLLM
            llm = LocalLLM()
            if hasattr(llm, 'is_available') and callable(llm.is_available):
                llm.is_available()
            checks["model"] = "✅ ready"
        except Exception as e:
            checks["model"] = f"❌ unavailable: {str(e)[:100]}"

        # Check vector store
        try:
            from src.vectorstore.chroma_db import ChromaVectorStore
            vs = ChromaVectorStore()
            count = vs.collection.count()
            checks["vectorstore"] = f"✅ ready ({count} docs)"
        except Exception as e:
            checks["vectorstore"] = f"❌ unavailable: {str(e)[:100]}"

        # If any critical checks failed, return 503
        if any("❌" in str(v) for v in checks.values()):
            return {"status": "not_ready", "checks": checks}, 503

        return {"status": "ready", "checks": checks}, 200

    logger.info(
        f"FastAPI app created host={config.api_host} port={config.api_port} log_level={config.log_level}"
    )
    return app


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        create_app(),
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower(),
    )
