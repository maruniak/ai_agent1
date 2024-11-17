from llama_index.core.tools import FunctionTool # allow function use
from llama_index.llms.ollama import Ollama      # the Ollama library
from llama_index.core.agent import ReActAgent   # the agent library
import wikipedia                                # the wikipedia library

# define sample Tools
def evaluate(inp: str) -> str:
    """evaluate an expression given in Python syntax"""
    return eval(inp)

def wikipedia_lookup(q: str) -> str:
    """Look up a query in Wikipedia and return the result"""
    return wikipedia.page(q).summary

eval_tool = FunctionTool.from_defaults(fn=evaluate)
wikipedia_lookup_tool = FunctionTool.from_defaults(fn=wikipedia_lookup)

# initialize llm
llm = Ollama(
    model = "gemma2:2b",
    request_timeout=120.0,
    temperature=0
)

agent = ReActAgent.from_tools([eval_tool, wikipedia_lookup_tool], 
                                llm=llm, verbose=True)

# A crude workflow
def kids_query(subject):
    prompts = [f"""Find out about {subject} and summarize your findings 
                   in about 100 words and respond with that summary""",
               """Re-write your summary so that it is suitable for a 
                  7-year old reader"""]

    for p in prompts:
        response = agent.chat(p)

    return response # return the last response

print(kids_query("Madrid"))