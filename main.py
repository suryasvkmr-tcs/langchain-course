from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_ollama import ChatOllama
load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Friends is an American television sitcom created by David Crane and Marta Kauffman, which aired on NBC from September 22, 1994, to May 6, 2004, lasting ten seasons. With an ensemble cast starring Jennifer Aniston, Courteney Cox, Lisa Kudrow, Matt LeBlanc, Matthew Perry, and David Schwimmer, the show revolves around six friends in their 20s and early 30s who live in Manhattan, New York City. The original executive producers were Kevin S. Bright, Kauffman, and Crane.

Kauffman and Crane began developing Friends under the working title Insomnia Cafe between November and December 1993. They presented the idea to Bright, and together they pitched a seven-page treatment of the show to NBC. After several script rewrites and changes, including title changes to Six of One and Friends Like Us, the series was finally named Friends. Filming took place at Warner Bros. Studios in Burbank, California.

All ten seasons of Friends ranked within the top ten of the final television season ratings; ultimately reaching the number 1 spot in its eighth season. The series finale aired on May 6, 2004, and was watched by around 52.5 million American viewers, making it the fifth-most-watched series finale in American television history and the most-watched television episode of the 2000s. Friends received acclaim throughout its run, becoming one of the most popular and highest-grossing television shows of all time. The show's success led to a spin-off series, Joey, and a reunion special, Friends: The Reunion.
    """

    summary_template = """
    given the information {information} about a perosn I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables = ["information"], template=summary_template
    )
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
