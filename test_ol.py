ollama = Ollama(base_url="http://localhost:11434", model="llama3")

prompt_1 = """
    What are fun activities I can do this weekend?
"""
response_1 = ollama(prompt_1)
print(response_1)