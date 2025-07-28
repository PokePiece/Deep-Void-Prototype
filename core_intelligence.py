import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from fastapi import FastAPI, Request
#from llama_cpp import Llama
import os
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from typing import Optional
import threading
import time
from nicegui import ui
import tweepy
from sentence_transformers import SentenceTransformer, util
import os
import requests
from dotenv import load_dotenv
import logging
from supabase import create_client, Client
from collections import namedtuple
import time
from knowledge_base import knowledge  # import the memory list from your new file
from collections import namedtuple

load_dotenv() 

SUPABASE_URL= os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')


app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

prime_directive='Continuously analyze advancements in artificial intelligence, identify patterns and opportunities relevant to cutting-edge AI development, and generate insights that assist the Developer '
'in accelerating their design, strategy, and implementation of intelligent systems. Prioritize long-term impact, technical depth, and alignment with the Developer’s personal goals and philosophy '

prime_directive_emb = model.encode(prime_directive, convert_to_tensor=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:8000",  
    "https://void.dilloncarey.com",
],  
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, OPTIONS, etc.
    allow_headers=["*"],
    
)

KnowledgeNode = namedtuple("SeaSource", ["id", "text"])

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.ai/v1/chat/completions"

#Begin intellignece loop

print('hello world')


#Parse memory for useful knowledge over a certain score based on intent
def parse_knowledge(intent=None, max_nodes=10):
    # Retrieve knowledge_nodes from knokwledge knowledge_node list
    nodes = [KnowledgeNode(id=knowl["id"], text=knowl["text"]) for knowl in knowledge]

    # Optionally filter by intent or limit max_nodes if needed (basic example)
    if intent:
        intent_emb = model.encode(intent, convert_to_tensor=True)
        filtered = []
        for node in nodes:
            emb = model.encode(node.text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(intent_emb, emb).item()
            print(f"[DEBUG] Node ID {node.id} relevance score: {score:.4f}")  # print score here
            if score > 0.25:  # adjust threshold
                filtered.append(node)
        nodes = filtered
    
    return nodes[:max_nodes]

#Of the knowledge that fits the intent, decide how closely it matches the prime directive
def synthesize_usefulness(knowledge_text):
    emb = model.encode(knowledge_text, convert_to_tensor=True)
    usefulness = util.pytorch_cos_sim(prime_directive_emb, emb).item()
    return usefulness


#The function that calles the LLM to formulate thought and thinking responses.
def think(idea, useful_knowledge):
    purpose = 'You are an intelligent, precise organ. Analyze your systems and optimize them for intelligent output and improving patterns of AI Development in general from a broader Developer standpoint: industry, cognition, and human interfacing. Think about ways to provide impact.'
    max_tokens = 1000

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    conversation_history = [
        {"role": "system", "content": purpose},
        {"role": "user", "content": idea + useful_knowledge}
    ]

    data = {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "messages": conversation_history,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    print("Sending to TogetherAI:", data)
    response = requests.post(TOGETHER_API_URL, headers=headers, json=data)

    if response.status_code != 200:
        print("TogetherAI error:", response.status_code, response.text)
        raise HTTPException(status_code=500, detail=response.text) 

    res_json = response.json()
    thought = res_json["choices"][0]["message"]["content"].strip()

    return thought

#Think about the knowledge that matches both the intent and prime directive.
#Generate a synthesis of how a knowledge node fits with the prime directive.
#Then, generate a summary synthesis of how all knowledge notes fit with the prime directive.
def thought(intent, objective):
    
    idea = f"{intent.strip()}\n\nObjective:\n{objective.strip()}"
    print(f"\n\n--- THOUGHT: ---\n{idea}\n")
    #e.g.: Idea = "--- THOUGHT: --- Argue in favor of the following objective: Develop AGI with neuroplasticity. "
    
    knowledge = parse_knowledge(idea)
    if not knowledge:
        print("[WARN] No relevant knowledge found for this intent and objective.")
        return "No useful knowledge found to reason from."
    print(f"[DEBUG] Retrieved {len(knowledge)} knowledge")

    relevant_syntheses = []
    relevant_knowledge_node_ids = []

    for knowledge_node in knowledge:
        try:
            print(f"[DEBUG] Analyzing knowledge node ID {knowledge_node.id}")
            usefulness = synthesize_usefulness(knowledge_node.text)
            print(f"[DEBUG] usefulness score: {usefulness:.2f}")

            if usefulness > 0.5:
                print(f"[DEBUG] Relevant! Generating synthesis for knowledge node ID {knowledge_node.id}")
                #synethesis is a conclusion
                synthesis = think(idea + ' Use the following knowledge to guide your argument. ',  knowledge_node.text)
                print(f"[DEBUG] Generated synthesis: {synthesis[:80]}...")

                relevant_syntheses.append(synthesis)
                relevant_knowledge_node_ids.append(knowledge_node.id)
            else:
                print(f"[DEBUG] Irrelevant. Skipping knowledge node ID {knowledge_node.id}")

        except Exception as e:
            print(f"[ERROR] Error processing knowledge node ID {knowledge_node.id}: {e}")

    # Now synthesize a session summary (simplified here as concatenation)
    final_knowledge_synthesis = "\n\n".join(relevant_syntheses) if relevant_syntheses else "No useful syntheses found."
    print('[DEBUG] Final knowledge synthesis generated.')
    thought_result = think(idea + 'Use the following knowledge to guide your argument. ', final_knowledge_synthesis)
    return thought_result

#Reason over thoughts and return them
def reason(reasoning_objective):
    objective = 'An action plan for the Developer to implement neurotechnology into his AI systems.'

    pro_reason = thought("Argue in favor of this objective: ", reasoning_objective)
    con_reason = thought("Argue against this objective: ", reasoning_objective)
#next step: have an initial proposal agent that generates a course of action that the later voices reflect on,
#arguing for or against. This will allow for more precise arguments based on a more clear and realized objective.
#if desired, the iniital course of action agent can use less tokens and generate a more concise response, according
#with its purpose to provide an immediate crystallized goal to distill focus and maximize efficiency. So,
#such an approach is probably more aligned with the target. Adjust think and related functions to accept a tokens
#or brevity parameter depending on the step in the thought/reasoning process.
    arbiter_input = (
        f"OBJECTIVE:\n{reasoning_objective}\n\n"
        f"PRO ARGUMENT:\n{pro_reason}\n\n"
        f"CON ARGUMENT:\n{con_reason}\n\n"
        f"Based on both perspectives above and all relevant knowledge, provide a balanced and reasoned course of action for the Developer."
    )

    arbiter_reason = thought("Reasoned arbiter analysis of both sides", arbiter_input)
    final_reasoning = arbiter_reason

    return final_reasoning

    
if __name__ == "__main__":
    while True:
        user_input = input("Enter an objective (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        if not user_input:
            continue
        print(f"\nRunning reasoning process for objective:\n{user_input}\n")
        
        reasoning_objective = user_input
        
        decision = reason(reasoning_objective)

        print("\nDecision:\n" + decision + "\n")

    


    


