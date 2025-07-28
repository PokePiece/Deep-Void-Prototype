

from typing import List, Dict
import uuid

knowledge: List[Dict] = []

def add_knowledge(text: str, category: str = "general", tags: List[str] = None):
    node = {
        "id": str(uuid.uuid4()),
        "text": text,
        "category": category,
        "tags": tags or []
    }
    knowledge.append(node)

def get_knowledge(category: str = None, tag: str = None) -> List[Dict]:
    results = knowledge
    if category:
        results = [m for m in results if m["category"] == category]
    if tag:
        results = [m for m in results if tag in m["tags"]]
    return results

def clear_knowledge():
    knowledge.clear()


# Initial core memories
add_knowledge(
    "Developer is developing a neuronic intelligence interface known as the Webtrix. Structured as a type of Metaverse, the app is a 3D embodiment of the real world mapped onto a virtual domain to embody the web in a physical presence, grounding intelligence in reality.",
    category="project",
    tags=["webtrix", "metaverse", "neuronic"]
)

add_knowledge(
    "The Developer has a cloud intelligence superstructure known as the Void that allows for the crystallized pool of intelligence to be accessed anywhere on the globe. He also has an embedded AGI system made to mimic human consciousness and productivity at all levels structured in C++, C, and meta-languages to enable superior control.",
    category="infrastructure",
    tags=["void", "AGI", "embedded"]
)

add_knowledge(
    "Seek to identify new industry trends and emerging technologies in the world of tech. Especially target neurotechnology as it relates to AI and intelligence, cloud and embedded AI systems, and more. Analyze existing knowledge for new ideas along with potential paths of improvement.",
    category="strategy",
    tags=["neurotech", "trend", "innovation"]
)
