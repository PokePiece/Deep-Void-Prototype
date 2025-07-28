

#This has the system run as a CLI


def begin_intelligence():
    system_prompt = "You are an intelligent system. You will perform intelligent tasks and analysis continuously."
    print("Starting CLI chat. Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break
        reason = call_reason(system_prompt, prompt)
        print("AI:", reason)

def call_reason(system_prompt: str, prompt: str, max_tokens: int = 512) -> str:
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
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
    return res_json["choices"][0]["message"]["content"].strip()






if __name__ == "__main__":
    begin_intelligence()
    
    




#printing reason
if __name__ == "__main__":
    print('Final reasoning: ' + reason())