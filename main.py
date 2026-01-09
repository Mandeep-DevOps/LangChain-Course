from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Arnold Alois Schwarzenegger[b] (born July 30, 1947) is an Austrian and American actor, businessman, former politician, and former professional bodybuilder, known for his roles in high-profile action films. He served as the 38th governor of California from 2003 to 2011.[3]

Schwarzenegger began lifting weights at age 15 and won the Mr. Universe title aged 20, and subsequently the Mr. Olympia title seven times. He is tied with Phil Heath for the joint-second number of all-time Mr. Olympia wins, behind Ronnie Coleman and Lee Haney, who are joint-first with eight wins each. Nicknamed the "Austrian Oak" in his bodybuilding days, he is regarded as one of the greatest bodybuilders of all time.[4][5] He has written books and articles about bodybuilding, including the autobiographical Arnold: The Education of a Bodybuilder (1977) and The New Encyclopedia of Modern Bodybuilding (1998).[6][7] The Arnold Sports Festival, the second-most prestigious bodybuilding event after the Mr. Olympia competition, is named after him.[8] He appeared in the bodybuilding documentary Pumping Iron (1977), which set him on his way to a career in films.[9]

After retiring from bodybuilding, Schwarzenegger gained worldwide fame as a Hollywood action star, with his breakthrough in the sword and sorcery epic Conan the Barbarian (1982),[10] a box-office success with a sequel in 1984.[11] After playing the title character in the science fiction film The Terminator (1984), he starred in Terminator 2: Judgment Day (1991) and three other sequels from 2003 to 2019. His other successful action films included Commando (1985), The Running Man (1987), Predator (1987), Total Recall (1990), and True Lies (1994), in addition to comedy films such as Twins (1988), Kindergarten Cop (1990) and Jingle All the Way (1996).[12] At the height of his career, Schwarzenegger was known for his rivalry with Sylvester Stallone.[13] Films in which he has appeared have grossed over $5.4 billion worldwide.[14] He is the founder of the film production company Oak Productions.[15]

As a registered member of the Republican Party, Schwarzenegger chaired the President's Council on Physical Fitness and Sports during most of the George H. W. Bush administration. In 2003, he was elected governor of California in a special recall election to replace Gray Davis, the governor at the time. He received 48.6 percent of the vote, 17 points ahead of the runner-up, Cruz Bustamante of the Democratic Party. He was sworn in on November 17 to serve the remainder of Davis' term, and was reelected in the 2006 gubernatorial election with an increased vote share of 55.9 percent to serve a full term.[16] In 2011, he reached his term limit as governor and returned to acting. As of 2025, Schwarzenegger and insurance commissioner Steve Poizner are the last Republicans to win or hold statewide office in California, having both won their respective elections in 2006.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short StackSummary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatOllama(model="gemma3:270m", temperature=0)
    llm = ChatOpenAI(model="gpt-5", temperature=0)
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
