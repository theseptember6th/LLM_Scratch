from langchain_core.prompts import ChatPromptTemplate


# This didnt work
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


chat_template = ChatPromptTemplate(
    [
        SystemMessage(content="You are a helpful {domain} expert"),
        HumanMessage(content="Explain in simple terms,what is {topic}"),
    ]
)

prompt = chat_template.invoke({"domain": "cricket", "topic": "Dusra"})

print(prompt)
