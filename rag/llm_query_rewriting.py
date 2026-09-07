import torch
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()



def rewrite_query(user_query):
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    
    system_prompt = """
    You are an expert movie researcher. Your job is to take a vague user 
    request and rewrite it into a descriptive search query that focuses 
    on plot themes, genres, and character descriptions. 
    Output ONLY the rewritten search string.
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Rewrite this for a movie search engine: {user_query}"}
        ],
        temperature=0.7,
        seed=42
    )
    return response.choices[0].message.content

query = "Weary Wolverine cares for an ailing Professor X in a hideout on the Mexican border"

if __name__=="__main__":
    print(rewrite_query(query))