from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
import os

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Healthcare-specific Tavily search tool
tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=False,
    include_domains=[
        "mayoclinic.org",         # Trusted medical information
        "webmd.com",              # Patient-centered medical content
        "medlineplus.gov",        # NIH health information
        "nih.gov",                # U.S. National Institutes of Health
        "ncbi.nlm.nih.gov",       # Research articles and PubMed data
        "clevelandclinic.org",    # Medical insights and resources
        "health.harvard.edu",     # Harvard Medical School publications
        "who.int",                # World Health Organization
        "cdc.gov"                 # Centers for Disease Control and Prevention
    ],
    exclude_domains=["socialmediahealthblog.com", "randomhealthbuzz.net"]
)
