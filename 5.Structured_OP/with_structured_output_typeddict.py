from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()
model = ChatGroq(
    model= "llama-3.1-8b-instant"
)

#schema
class review (TypedDict):
    summary :str
    sentiment : str

structured_model = model.with_structured_output(review)
result = structured_model.invoke(
    "Moto G85"
)
print(result)