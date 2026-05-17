from fastapi import FastAPI
from pydantic import BaseModel
from src.blog_writing_agent import run

app = FastAPI()


class BlogRequest(BaseModel):
    topic: str


@app.post("/generate")
def generate_blog(req: BlogRequest):

    result = run(req.topic)

    return {
        "blog": result["final"]
    }
