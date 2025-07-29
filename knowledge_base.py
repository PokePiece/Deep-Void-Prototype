from typing import List, Dict
import uuid
import json
import os

KNOWLEDGE_FILE = "knowledge.json"

knowledge: List[Dict] = []

def load_knowledge():
    global knowledge
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            knowledge = json.load(f)
    else:
        knowledge = []

def save_knowledge():
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2)

def add_knowledge(text: str, category: str = "general", tags: List[str] = None):
    node = {
        "id": str(uuid.uuid4()),
        "text": text,
        "category": category,
        "tags": tags or []
    }
    knowledge.append(node)
    save_knowledge()

def get_knowledge(category: str = None, tag: str = None) -> List[Dict]:
    results = knowledge
    if category:
        results = [m for m in results if m.get("category") == category]
    if tag:
        results = [m for m in results if tag in m.get("tags", [])]
    return results

def clear_knowledge():
    global knowledge
    knowledge.clear()
    save_knowledge()
    
def update_knowledge(node_id: str, text: str = None, category: str = None, tags: List[str] = None) -> bool:
    """
    Update a knowledge node by its id.
    Returns True if updated, False if not found.
    """
    for node in knowledge:
        if node["id"] == node_id:
            if text is not None:
                node["text"] = text
            if category is not None:
                node["category"] = category
            if tags is not None:
                node["tags"] = tags
            save_knowledge()
            return True
    return False

load_knowledge()

save_knowledge()


