from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


from third_parties.linkedin import scrape_linkefin_profile
from third_parties.gist import get_gist_data
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent


def ice_break(name: str) -> str:
    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two intresting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # gist_data = get_gist_data(
    #     "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"
    # )

    # print(chain.run(information=gist_data))

    linkedin_profile_url = linkedin_lookup_agent(name="Enrico Medeiros")

    # print(linkedin_profile_url)

    linkedin_data = scrape_linkefin_profile(linkedin_profile_url)

    result = (chain.run(information=linkedin_data))
    print(result)
    return result


if __name__ == "__main__":
    print("Hello Langchain!")

    ice_break(name="Enrico Medeiros")
