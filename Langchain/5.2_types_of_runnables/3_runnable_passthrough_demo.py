from langchain_core.runnables import RunnablePassthrough

passthrough = RunnablePassthrough()
print(passthrough.invoke(2))  # o/p : 2
print(passthrough.invoke({"name": "kristal"}))  # same dictionary as o/p
