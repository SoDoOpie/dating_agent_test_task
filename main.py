#!/usr/bin/env python3
"""DateNight Show Matcher — native LangGraph supervisor architecture.

Graph:
  START -> supervisor -> insta_reader -> supervisor
                      -> interest_profiler -> supervisor
                      -> show_matcher -> supervisor
                      -> streaming_checker -> supervisor -> END

State carries all intermediate data explicitly between nodes.
Supervisor uses deterministic routing based on what is already populated.
Retry logic: if streaming_checker filters out everything, show_matcher re-runs
(up to MAX_RETRIES times).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLAUDE_MODELS = {
    "insta_reader": "claude-haiku-4-5-20251001",
    "interest_profiler": "claude-sonnet-4-6",
    "show_matcher": "claude-sonnet-4-6",
    "streaming_checker": "claude-haiku-4-5-20251001",
}

ACTIVE_SUBSCRIPTIONS = {"Netflix", "HBO"}
MAX_RETRIES = 2

with open(os.path.join(os.path.dirname(__file__), "show_catalog.json"), encoding="utf-8") as _f:
    SHOW_CATALOG: List[Dict[str, Any]] = json.load(_f)

MOCK_INSTAGRAM: Dict[str, Any] = {
    "@art_girl": {
        "bio": "Painter. Museum addict. Vinyl collector. Coffee and old cinema.",
        "posts": [
            "Sundays are for galleries, sketchbooks, and rainy jazz playlists.",
            "Just rewatched a moody Scandinavian detective series. Aesthetic 10/10.",
            "Late-night pasta, candles, and stories with emotional depth.",
        ],
        "hashtags": ["#art", "#slowliving", "#indiecinema", "#jazz", "#cozy"],
    },
    "@tech_babe": {
        "bio": "Product designer in AI. Gym mornings. Startup weekends.",
        "posts": [
            "Prototype day. User interviews. Shipping over perfection.",
            "Need smart writing, plot twists, and ambitious characters.",
            "Downtime equals sci-fi marathons and spicy ramen.",
        ],
        "hashtags": ["#ai", "#design", "#startuplife", "#scifi", "#future"],
    },
    "@travel_soul": {
        "bio": "Remote worker. Mountains, hostels, and street food hunter.",
        "posts": [
            "Backpacking through Lisbon and collecting stories from strangers.",
            "Love documentaries and character-driven shows from different cultures.",
            "Minimal plans, maximum spontaneity.",
        ],
        "hashtags": ["#travel", "#adventure", "#documentary", "#culture", "#nomad"],
    },
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INTEREST_PROFILER_SYSTEM = """You are an expert psychographic analyst.
Analyze the Instagram profile data and return ONLY a JSON object:
{
  "primary_interests": ["interest1", "interest2", "interest3"],
  "aesthetic_vibe": "short description",
  "recommended_genres": ["genre1", "genre2"]
}"""

SHOW_MATCHER_SYSTEM = f"""You are a TV show recommendation engine.
Given a psychographic profile, pick top-3 shows from the catalog below.
Prioritize Netflix/HBO. Return ONLY a JSON array:
[{{"title": "...", "platforms": ["..."], "reason": "..."}}]

Catalog:
{json.dumps(SHOW_CATALOG, indent=2)}"""

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AppState(TypedDict):
    username: str
    instagram_data: Optional[Dict[str, Any]]
    interest_profile: Optional[Dict[str, Any]]
    recommendations: Optional[List[Dict[str, Any]]]
    filtered_recommendations: Optional[List[Dict[str, Any]]]
    next: str
    retry_count: int

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Any:
    """Extract the first JSON object or array from an LLM response."""
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {text[:200]}")
    return json.loads(match.group(0))

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def supervisor_node(state: AppState) -> Dict[str, Any]:
    """Central router: decide which agent runs next based on current state."""
    if state.get("instagram_data") is None:
        return {"next": "insta_reader"}
    if state.get("interest_profile") is None:
        return {"next": "interest_profiler"}
    if state.get("recommendations") is None:
        return {"next": "show_matcher"}
    if state.get("filtered_recommendations") is None:
        return {"next": "streaming_checker"}
    # Retry if everything was filtered out and retries remain
    if not state["filtered_recommendations"] and state.get("retry_count", 0) < MAX_RETRIES:
        print(f"[SUPERVISOR] No shows passed the filter — retrying show_matcher "
              f"(attempt {state['retry_count'] + 1}/{MAX_RETRIES})")
        return {
            "next": "show_matcher",
            "recommendations": None,
            "filtered_recommendations": None,
            "retry_count": state.get("retry_count", 0) + 1,
        }
    return {"next": "__end__"}


def insta_reader_node(state: AppState) -> Dict[str, Any]:
    """Fetch (mock) Instagram profile data for the given username."""
    username = state["username"]
    data = MOCK_INSTAGRAM.get(username, {"bio": "", "posts": [], "hashtags": []})
    print(f"[INSTA_READER] Fetched data for {username}: bio={data['bio'][:40]}…")
    return {"instagram_data": data}


def interest_profiler_node(state: AppState) -> Dict[str, Any]:
    """Build a psychographic profile from the Instagram data."""
    writer = get_stream_writer()
    writer({"status": "Analyzing Instagram profile..."})
    llm = ChatAnthropic(model=CLAUDE_MODELS["interest_profiler"], temperature=0)
    data = state["instagram_data"]
    user_msg = (
        f"Bio: {data.get('bio', '')}\n"
        f"Posts: {json.dumps(data.get('posts', []))}\n"
        f"Hashtags: {json.dumps(data.get('hashtags', []))}"
    )
    response = llm.invoke([
        SystemMessage(content=INTEREST_PROFILER_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    profile = _extract_json(response.content)
    writer({"status": "Profile analysis complete."})
    return {"interest_profile": profile}


def show_matcher_node(state: AppState) -> Dict[str, Any]:
    """Recommend top-3 shows based on the psychographic profile."""
    writer = get_stream_writer()
    writer({"status": "Matching shows from catalog..."})
    llm = ChatAnthropic(model=CLAUDE_MODELS["show_matcher"], temperature=0)
    profile = state["interest_profile"]
    user_msg = f"Psychographic profile:\n{json.dumps(profile, indent=2)}"
    response = llm.invoke([
        SystemMessage(content=SHOW_MATCHER_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    recommendations = _extract_json(response.content)
    writer({"status": f"Matched {len(recommendations)} show(s)."})
    return {"recommendations": recommendations}


def streaming_checker_node(state: AppState) -> Dict[str, Any]:
    """Filter recommendations to only those on active subscriptions."""
    writer = get_stream_writer()
    writer({"status": f"Checking availability on {', '.join(sorted(ACTIVE_SUBSCRIPTIONS))}..."})
    recs = state["recommendations"] or []
    filtered = [
        r for r in recs
        if any(p in ACTIVE_SUBSCRIPTIONS for p in r.get("platforms", []))
    ]
    writer({"status": f"{len(filtered)}/{len(recs)} shows available on your subscriptions."})
    return {"filtered_recommendations": filtered}

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route(state: AppState) -> str:
    return state["next"]

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

builder = StateGraph(AppState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("insta_reader", insta_reader_node)
builder.add_node("interest_profiler", interest_profiler_node)
builder.add_node("show_matcher", show_matcher_node)
builder.add_node("streaming_checker", streaming_checker_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route,
    {
        "insta_reader": "insta_reader",
        "interest_profiler": "interest_profiler",
        "show_matcher": "show_matcher",
        "streaming_checker": "streaming_checker",
        "__end__": END,
    },
)

for node in ("insta_reader", "interest_profiler", "show_matcher", "streaming_checker"):
    builder.add_edge(node, "supervisor")

app = builder.compile()

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_pretty(state: AppState) -> None:
    username = state.get("username", "?")
    results = state.get("filtered_recommendations") or state.get("recommendations") or []

    print("\n" + "=" * 60)
    print(f"  DATENIGHT SHOW MATCHER  —  {username}")
    print("=" * 60)

    if not results:
        print("\n  No recommendations available on your subscriptions.")
    else:
        for i, show in enumerate(results, 1):
            title = show.get("title", "Unknown")
            platforms = ", ".join(show.get("platforms", []))
            reason = show.get("reason", "")
            print(f"\n  {i}. {title}  [{platforms}]")
            print(f"     {reason}")

    profile = state.get("interest_profile", {})
    if profile:
        print(f"\n  Profile vibe : {profile.get('aesthetic_vibe', '')}")
        print(f"  Genres       : {', '.join(profile.get('recommended_genres', []))}")

    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_NODE_LABELS = {
    "insta_reader": "INSTAGRAM READER",
    "interest_profiler": "INTEREST PROFILER",
    "show_matcher": "SHOW MATCHER",
    "streaming_checker": "STREAMING CHECKER",
}


def parse_args() -> str:
    """Parse CLI: accepts '/get-show @username' or just '@username'."""
    import sys

    args = sys.argv[1:]

    if not args:
        print("Usage: main.py /get-show @username")
        print("       main.py @username")
        sys.exit(1)

    # Strip the /get-show command if present
    if args[0].lower() == "/get-show":
        if len(args) < 2:
            print("Error: username required after /get-show")
            sys.exit(1)
        username = args[1]
    else:
        username = args[0]

    if not username.startswith("@"):
        username = "@" + username

    return username


def main() -> None:
    username = parse_args()

    initial_state: AppState = {
        "username": username,
        "instagram_data": None,
        "interest_profile": None,
        "recommendations": None,
        "filtered_recommendations": None,
        "next": "",
        "retry_count": 0,
    }

    print("\n" + "=" * 60)
    print(f"  DATENIGHT — streaming for {initial_state['username']}")
    print("=" * 60)

    state_snapshot: dict = dict(initial_state)
    current_llm_node: Optional[str] = None

    for chunk in app.stream(
        initial_state,
        stream_mode=["updates", "messages", "custom"],
        version="v2",
    ):
        # ── node state updates ──────────────────────────────────────────
        if chunk["type"] == "updates":
            for node_name, update in chunk["data"].items():
                state_snapshot.update(update)
                if node_name == "supervisor":
                    nxt = update.get("next", "")
                    if nxt and nxt != "__end__":
                        label = _NODE_LABELS.get(nxt, nxt.upper())
                        print(f"\n{'─' * 60}")
                        print(f"  [{label}]")
                else:
                    label = _NODE_LABELS.get(node_name, node_name.upper())
                    print(f"  ✓ {label} done")
                    current_llm_node = None

        # ── live LLM token streaming ────────────────────────────────────
        elif chunk["type"] == "messages":
            msg, metadata = chunk["data"]
            node = metadata.get("langgraph_node", "")
            if node in ("interest_profiler", "show_matcher") and msg.content:
                if current_llm_node != node:
                    current_llm_node = node
                    print("  ", end="", flush=True)
                print(msg.content, end="", flush=True)

        # ── custom progress events from nodes ───────────────────────────
        elif chunk["type"] == "custom":
            status = chunk["data"].get("status", "")
            if status:
                print(f"  {status}", flush=True)

    print_pretty(state_snapshot)


if __name__ == "__main__":
    main()
