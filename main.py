from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


load_dotenv()

def main():
    print("Hello, World!")
    information = '''Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of April 2026, Forbes estimates his net worth to be US$809 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.'''

    summary_template = """
    given information {information} about a person I want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    summary_prompt_template = ChatPromptTemplate.from_template(summary_template)  # ✅ fix 1

    llm = ChatOllama(model="gemma3:270m")  # ✅ fix 2
    
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":  # ✅ fix 3
    main()