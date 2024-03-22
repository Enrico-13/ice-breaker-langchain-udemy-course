from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from output_parsers import person_intel_parser, PersonIntel
from third_parties.linkedin import scrape_linkefin_profile
from third_parties.gist import get_gist_data
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent


def ice_break(name: str) -> PersonIntel:
    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two intresting facts about them
    3. A topic that may interest them
    4. 2 creative ice breakers to open a conversation with them
    \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    gist_data = get_gist_data(
        "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"
    )

    # print(chain.run(information=gist_data))

    # linkedin_profile_url = linkedin_lookup_agent(name)

    # print(linkedin_profile_url)

    # linkedin_data = scrape_linkefin_profile(linkedin_profile_url)

    # result = chain.run(information=linkedin_data)
    result = chain.run(information=gist_data)

    gist_data["profile_pic_url"] = 'https://media.licdn.com/dms/image/C4D03AQGlv35ItbkHBw/profile-displayphoto-shrink_400_400/0/1610187870291?e=1716422400&v=beta&t=Hb9E0lFt9Ivhz7JhpoZ4h-Jf0KMDpncL21GJ2wrQCT8'

    # return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")
    return person_intel_parser.parse(result), gist_data.get("profile_pic_url")


if __name__ == "__main__":
    print("Hello Langchain!")

    result = ice_break(name="Enrico Medeiros")
