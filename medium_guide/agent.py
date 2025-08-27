import os
from typing import TypedDict, List 

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv() 


from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Creating Agent's Memory

# Define a TypedDict named 'State' to represent a structured dictionary
class State(TypedDict):
    text: str               # stores the original input text
    classification: str     # represents the classification result (e.g., category label)
    entities: List[str]     # holds a list of extracted entities (e.g., named entities)
    summary: str            # stores a summarized version of the text

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Adding Agent's Capabilities

# 분류
def classification_node(state: State):
   """
   Classify the text into one of predefined categories.
   
   Parameters:
       state (State): The current state dictionary containing the text to classify
       
   Returns:
       dict: A dictionary with the "classification" key containing the category result
       
   Categories:
       - News: Factual reporting of current events
       - Blog: Personal or informal web writing
       - Research: Academic or scientific content
       - Technology: Content related to technology and its advancements
       - Other: Content that doesn't fit the above categories
   """

   # define a prompt template
   prompt = PromptTemplate(
       input_variables=["text"],
       template="Classify the following text into one of the categories: News, Blog, Research, technology, Other.\n\nText: {text}\n\nCategory:"
   )

   # state 로 부터 input text를 포함하는 프롬프트 생성
   message = HumanMessage(content=prompt.format(text=state["text"]))

    # Invoke the language model to classify the text based on the prompt
   classification = llm.invoke([message]).content.strip()

   # 딕셔너리 내 classification 반환
   return {"classification": classification}

# Entity 추출 
def entity_extraction_node(state: State):
  # Function to identify and extract named entities from text
  # Organized by category (Person, Organization, Location)
  
  # Create template for entity extraction prompt
  # Specifies what entities to look for and format (comma-separated)
  prompt = PromptTemplate(
      input_variables=["text"],
      template="Extract all the entities (Person, Organization, Location, Technologies) from the following text. Provide the result as a comma-separated list.\n\nText: {text}\n\nEntities:"
  )
  
  # Format the prompt with text from state and wrap in HumanMessage
  message = HumanMessage(content=prompt.format(text=state["text"]))
  
  # Send to language model, get response, clean whitespace, split into list
  entities = llm.invoke([message]).content.strip().split(", ")
  
  # Return dictionary with entities list to be merged into agent state
  return {"entities": entities}

# 요약
def summarize_node(state: State):
    # Create a template for the summarization prompt
    # This tells the model to summarize the input text in one sentence
    summarization_prompt = PromptTemplate.from_template(
        """Summarize the following text in one short sentence.
        
        Text: {text}
        
        Summary:"""
    )
    
    # Create a chain by connecting the prompt template to the language model
    # The "|" operator pipes the output of the prompt into the model
    chain = summarization_prompt | llm
    
    # Execute the chain with the input text from the state dictionary
    # This passes the text to be summarized to the model
    response = chain.invoke({"text": state["text"]})
    
    # Return a dictionary with the summary extracted from the model's response
    # This will be merged into the agent's state
    return {"summary": response.content}


# Agent 구조
workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarize_node)

# Add edges to the graph
workflow.set_entry_point("classification_node") # Set the entry point of the graph
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# Compile the graph
app = workflow.compile()

# Visualize the graph and save it as a PNG file
# Note: This requires graphviz to be installed on your system.
# For macOS: brew install graphviz
# For Debian/Ubuntu: sudo apt-get install graphviz
# Then, install the Python wrapper: uv pip install pygraphviz
try:
    image_data = app.get_graph().draw_mermaid_png()
    with open("agent_workflow.png", "wb") as f:
        f.write(image_data)
    print("\nAgent workflow visualization saved as agent_workflow.png")
except Exception as e:
    print(f"\nCould not visualize the graph. Make sure graphviz and pygraphviz are installed. Error: {e}")

# Define a sample text about Anthropic's MCP to test our agent
sample_text = """
Anthropic's MCP (Model Context Protocol) is an open-source powerhouse that lets your applications interact effortlessly with APIs across various systems.
"""

# Create the initial state with our sample text
state_input = {"text": sample_text}

# Run the agent's full workflow on our sample text
result = app.invoke(state_input)

# Print each component of the result:
# - The classification category (News, Blog, Research, or Other)
print("Classification:", result["classification"])

# - The extracted entities (People, Organizations, Locations)
print("\nEntities:", result["entities"])

# - The generated summary of the text
print("\nSummary:", result["summary"])