# ==========================================================
# Multi-Turn AI Tutor
# LangChain + Hugging Face + Function Calling Simulation
# Model: Qwen/Qwen2.5-3B-Instruct
# ==========================================================

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage, AIMessage
import wikipedia

# ----------------------------------------------------------
# Load API Key from .env
# ----------------------------------------------------------
load_dotenv()

# ----------------------------------------------------------
# Create LLM Endpoint
# ----------------------------------------------------------
# This connects LangChain to the Hugging Face Inference API
# using Qwen 2.5 3B Instruct.
# ----------------------------------------------------------

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.3
)

chat_model = ChatHuggingFace(
    llm=llm
)

# ----------------------------------------------------------
# Tool 1 : Calculator
# ----------------------------------------------------------
# Receives a mathematical expression and evaluates it.
#
# Example:
# "25*8"
# Result:
# 200
# ----------------------------------------------------------

def calculator(expression):

    try:
        return str(eval(expression))

    except Exception:
        return "Invalid mathematical expression."


# ----------------------------------------------------------
# Tool 2 : Wikipedia Search
# ----------------------------------------------------------
# Searches Wikipedia and returns a short summary.
#
# Example:
# "Alan Turing"
#
# Returns:
# Alan Turing was a British mathematician...
# ----------------------------------------------------------

def wikipedia_tool(topic):

    try:

        return wikipedia.summary(
            topic,
            sentences=3
        )

    except Exception:

        return "Wikipedia could not find information."


# ----------------------------------------------------------
# Chat History
# ----------------------------------------------------------
# Stores previous messages.
#
# Multi-turn means:
#
# User: Who is Alan Turing?
# AI: ...
#
# User: When was he born?
# AI remembers previous context.
# ----------------------------------------------------------

chat_history = []


# ----------------------------------------------------------
# Main Chat Loop
# ----------------------------------------------------------

print("AI Tutor Started")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # ------------------------------------------------------
    # Tool Selection Logic
    # ------------------------------------------------------
    #
    # We ask the LLM:
    #
    # Which tool should be used?
    #
    # Allowed tools:
    # calculator
    # wikipedia
    # none
    #
    # ------------------------------------------------------

    tool_prompt = f"""

You are a tool selector.

Available tools:

1. calculator
   Use for mathematical calculations.

2. wikipedia
   Use for factual topics, people, places,
   inventions, concepts, history.

3. none
   Use if no tool is needed.

User Question:
{user_input}

Reply with only:

calculator
or
wikipedia
or
none

"""

    tool_response = chat_model.invoke(
        [HumanMessage(content=tool_prompt)]
    )

    selected_tool = tool_response.content.strip().lower()

    # ------------------------------------------------------
    # Execute Tool
    # ------------------------------------------------------

    tool_result = ""

    if "calculator" in selected_tool:

        tool_result = calculator(user_input)

    elif "wikipedia" in selected_tool:

        tool_result = wikipedia_tool(user_input)

    # ------------------------------------------------------
    # Final Prompt
    # ------------------------------------------------------
    #
    # If tool output exists,
    # provide it to the LLM.
    #
    # Otherwise answer normally.
    # ------------------------------------------------------

    final_prompt = f"""

You are a helpful AI Tutor.

Previous Conversation:
{chat_history}

User Question:
{user_input}

Tool Output:
{tool_result}

Answer the user clearly and concisely.

"""

    response = chat_model.invoke(
        [HumanMessage(content=final_prompt)]
    )

    print("\nTutor:", response.content)
    print()

    # ------------------------------------------------------
    # Store Conversation
    # ------------------------------------------------------

    chat_history.append(
        HumanMessage(content=user_input)
    )

    chat_history.append(
        AIMessage(content=response.content)
    )