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
import knowledge_base
from collections import namedtuple

load_dotenv() 

SUPABASE_URL= os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')


app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

knowledge_base.load_knowledge()

knowledge = knowledge_base.knowledge


prime_directive='Continuously analyze advancements in artificial intelligence, identify patterns and opportunities relevant to cutting-edge AI development, and generate insights that assist the Developer '
'in accelerating their design, strategy, and implementation of intelligent systems. Prioritize long-term impact, technical depth, and alignment with the Developer’s personal goals and philosophy '

prototype_prime_directive=''

prime_directive_emb = model.encode(prime_directive, convert_to_tensor=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:8000",  
    "https://void.dilloncarey.com",
],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
    
)

KnowledgeNode = namedtuple("SeaSource", ["id", "text"])

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.ai/v1/chat/completions"


#Parse memory for useful knowledge over a certain score based on intent
def parse_knowledge(intent=None, max_nodes=5):
    # Retrieve knowledge_nodes from knokwledge knowledge_node list
    nodes = []
    for knowl in knowledge:
        if "id" in knowl and "text" in knowl:
            nodes.append(KnowledgeNode(id=knowl["id"], text=knowl["text"]))


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


#The function that calls the LLM to formulate thought and thinking responses.
#Added tokens and brevity variables which will allow the think function to give long and concise output
def think(idea, purpose='', useful_knowledge='', tokens:int=1000, brevity:bool=False):
    subject = purpose or 'You are an intelligent, precise organ. Analyze your systems and optimize them for intelligent output and improving patterns of AI Development in general from a broader Developer standpoint: industry, cognition, and human interfacing. Think about ways to provide impact.'
    
    if brevity:
        print('being concise')
        concise_message='Give a concise review on the matter limited to a sharp paragraph.'
    else:
        concise_message=''
        
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

#Pass prime directive instead of purpose? Or both via concatenation.
    conversation_history = [
        {"role": "system", "content": prime_directive + subject},
        {"role": "user", "content": idea + useful_knowledge + concise_message}
    ]

    data = {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "messages": conversation_history,
        "max_tokens": tokens,
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
#Then, generate a summary synthesis of how all knowledge nodes fit with the prime directive.
def thought(intent, objective, tokens:int=1000, brevity:bool=False):
    
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
                synthesis = think(idea + ' Use the following knowledge to guide your argument. ', knowledge_node.text, 350, True)
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
    
    thought_result = think(idea + 'Use the following knowledge to guide your argument. ', final_knowledge_synthesis, tokens, brevity)
    return thought_result

#Reason over thoughts and return them
def reason(reasoning_objective):
    objective = 'Create AGI with true neuroplasticity for enhanced reasoning in legal domains.'

    initial_reason = thought("Develop an initial plan or approach via argument to realize this objective: ", reasoning_objective, 500, True)
    pro_reason = thought("Argue in favor of this plan/approach: ", initial_reason)
    con_reason = thought("Argue against this plan/approach: ", initial_reason)

    arbiter_input = (
        f"OBJECTIVE:\n{reasoning_objective}\n\n"
        f"PRO ARGUMENT:\n{pro_reason}\n\n"
        f"CON ARGUMENT:\n{con_reason}\n\n"
        f"Based on both perspectives above and all relevant knowledge, provide a balanced and reasoned course of action for the Developer."
    )

    arbiter_reason = thought("Reasoned arbiter analysis of both sides", arbiter_input)
    final_reasoning = arbiter_reason

    return final_reasoning

def chat(message):
    chat_guide = 'The Developer is chatting with you. Please respond in a technical, helpful, chat-like tone to respond to the prompt.'
    response = think(message, chat_guide)
    return response

def action(task):
    print('performing action')
    actions = ['reason', 'think', 'thought', 'synthesize_usefulness', 'parse_knowledge', 'chat', 'discussion']
    task_guide = ('You are now functioning as a task directing agent for the Developer. Given a prompt by the Developer, '
                  'you need to decide on an action to take based on its type and intent. You are to only reply with the selected action. '
                  'You are basically categorizing the nature of the prompt so another system can take an action. But really are deciding on '
                  'an action to take based on the prompt. The actions you can take are "chat", "think", and "reason". The overwhelming majority '
                  'of the time, assume the user is chatting with you, and select the chat action. Only if the user explicitly commands you to do one '
                  'of the other two things should you return those options. When giving your response for the action, return only your choice like '
                  '"chat", "reason", or "think", with nothing else. Again, without further context or unless explicitly prompted by the user in the '
                  'prompt simply return "chat". Now, the message from the Developer for you to classify is as follows:')
    
    action_type = think(task, task_guide)
    print("\nDecision:\n" + action_type + "\n")
    if 'chat' in action_type.lower():
        response = chat(task)

    print(response)
    return response


if __name__ == "__main__":
    while True:
        user_input = input("Enter a command or prompt (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        if not user_input:
            continue
        print(f"\nRunning processes for command or prompt:\n{user_input}\n")

        action(user_input)




'''
if __name__ == "__main__":
    while True:
        user_input = input("Enter an objective (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        if not user_input:
            continue
        print(f"\nRunning reasoning process for objective:\n{user_input}\n")
        
        objective = user_input
        
        reasoning = reason(objective)

        print("\nDecision:\n" + reasoning + "\n")
'''
    


    


