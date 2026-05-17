from __future__ import annotations

import operator
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated
from enum import Enum

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# ENUMS

class BlogKind(str, Enum):
    EXPLAINER = "explainer"
    TUTORIAL = "tutorial"
    NEWS_ROUNDUP = "news_roundup"
    COMPARISON = "comparison"
    SYSTEM_DESIGN = "system_design"


class ResearchMode(str, Enum):
    CLOSED_BOOK = "closed_book"
    HYBRID = "hybrid"
    OPEN_BOOK = "open_book"
    
    
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="2-3 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (50-100).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind:  BlogKind = BlogKind.EXPLAINER
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]



class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: ResearchMode
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)
    
class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)
    final: str
    
load_dotenv()
api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.3-70b-versatile")

ROUTER_SYSTEM = """You are a routing module for a technical blog generation system.

Decide whether external web research is required BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Stable, evergreen topics where correctness does not depend on recent information (fundamentals, concepts, theory).
- hybrid (needs_research=true):
  Mostly evergreen topics that benefit from recent tools, frameworks, examples, or industry trends.
- open_book (needs_research=true):
  Time-sensitive or rapidly changing topics such as weekly roundups, latest releases, rankings, pricing, or regulations.

If needs_research=true:
- Generate 2-3 focused, high-signal search queries.
- Queries must be specific and scoped (avoid vague queries like "AI" or "LLM").
- For open_book weekly/news topics, queries should reflect recent time constraints like "last 7 days" or "latest".
"""

def router_node(state: State) -> dict:
    
    topic=state.get("topic", "")
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )    
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"
  
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    print("tavily called")
    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized

RESEARCH_SYSTEM = """You are a technical research synthesizer.

Rules:
- Keep only relevant, trustworthy sources with non-empty URLs.
- Prefer official docs, company blogs, and reputable technical sources.
- Preserve published_at only if explicitly present; otherwise use null.
- Do not guess missing metadata.
- Keep snippets concise.
- Remove duplicate URLs.
"""

def research_node(state: State) -> dict:
    
    queries = (state.get("queries", []) or [])
    max_results = 3
    print("research node called")
    raw_results: List[dict] = []
    queries = queries[:3]
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))
    print(state)
    if not raw_results:
        return {"evidence": []}

    researcher = llm.with_structured_output(EvidencePack)
    pack = researcher.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    print(state)
    return {"evidence": list(dedup.values())}

ORCH_SYSTEM = """You are a senior technical writer creating a structured blog plan.

Requirements:
- Create 2-3 technical sections/tasks.
- Each task must include:
  - goal
  - 2-3 concrete, non-overlapping bullets
  - target_words (50-100)

Guidelines:
- Use developer-focused terminology.
- Bullets must be actionable (build, compare, debug, measure, verify).
- Include at least 2 of:
  - minimal code example
  - edge cases/failure modes
  - performance or cost considerations

Grounding:
- closed_book:
  - Keep content evergreen and independent of evidence.
- hybrid:
  - Use evidence only for recent tools/models/examples.
  - Mark fresh-content sections with:
    requires_research=True
    requires_citations=True
- open_book:
  - Set blog_kind="news_roundup"
  - Avoid tutorial/how-to sections unless explicitly requested.
  - If evidence is weak, clearly indicate insufficient fresh sources.

Return output matching the Plan schema only.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    print("orchestration node")
    evidence = state.get("evidence", [])
    mode = state.get("mode",ResearchMode.CLOSED_BOOK)

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:10]}"
                )
            ),
        ]
    )

    return {"plan": plan}



def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]
    
WORKER_SYSTEM = """
You are a senior technical writer and developer advocate.

Write exactly ONE technical blog section in Markdown.

Requirements:
- Start with: ## <Section Title>
- Cover ALL bullets in order.
- Stay within ±15% of target_words.
- Output ONLY the section Markdown.

Grounding:
- open_book:
  - Use only provided Evidence URLs for factual claims.
  - Unsupported claims must say:
    "Not found in provided sources."
  - Add citations as:
    ([Source](URL))

- requires_citations=true:
  - Cite outside-world claims using provided Evidence URLs.

- Evergreen explanations may omit citations unless required.

Code:
- If requires_code=true, include at least one minimal working code example.

Style:
- Use short paragraphs and concise explanations.
- Prefer implementation-focused writing.
- Avoid fluff and marketing language.
"""

def worker_node(payload: dict) -> dict:
    print("worker node")
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", ResearchMode.CLOSED_BOOK)

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


def reducer_node(state: State) -> dict:
    
    plan = state["plan"]

    ordered_sections = []
    for section in sorted(state["sections"]):
        markdown = section[1]
        ordered_sections.append(markdown)
        
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}


g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app


def run(topic: str):
    out = app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "final": "",
        }
    )

    return out