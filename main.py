"""Compatibility entrypoint for the FastAPI app.

Preferred command:
    uvicorn ai_backend.api.main:app --reload
"""

from ai_backend.api.main import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai_backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
