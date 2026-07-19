import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role":"system",
            "content":"You are a dumb AI that gives random answers that are unrelated"
        },
        {
            "role":"user",
            "content":"What is Groq AI. And how does it work locally"
        }
    ],
    temperature = 0.3
)

print(response.choices[0].message.content)
