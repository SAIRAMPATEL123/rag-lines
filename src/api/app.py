from time import perf_counter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from config.config import get_config
from src.api.routes import router


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title="RAG-Lines API",
        description="Retrieval-Augmented Generation System API",
        version="1.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = perf_counter()
        logger.info(f"HTTP {request.method} {request.url.path} started")
        response = await call_next(request)
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info(
            f"HTTP {request.method} {request.url.path} completed status={response.status_code} elapsed_ms={elapsed_ms:.2f}"
        )
        return response

    app.include_router(router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "rag-lines-api", "version": "1.1.0"}

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
