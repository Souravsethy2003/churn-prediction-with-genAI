from dotenv import load_dotenv
load_dotenv()

import os
import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ ENV ------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

# ------------------ FALLBACK ------------------
def _fallback(top_reasons_list: List[str], counts_str: str, summary_stats: str) -> str:
    return "⚠️ LLM unavailable. Check API key or internet."

# ------------------ PROMPT ------------------
_TEMPLATE = """
You are a senior product analyst and retention strategist.

Provide:
1. Executive Summary
2. Key Analytical Insights
3. Prioritized Process Improvements

Max 300 words.

Data:
Top reasons: {top_reasons_list}
Counts: {counts}
Summary: {summary_stats}
"""

_PROMPT = ChatPromptTemplate.from_template(_TEMPLATE)

# ------------------ LLM ------------------
def _create_llm():
    return ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=TEMPERATURE,
        google_api_key=API_KEY  # ✅ FIXED
    )

_llm = _create_llm()

# ------------------ CHAIN ------------------
_chain = _PROMPT | _llm | StrOutputParser()

# ------------------ MAIN FUNCTION ------------------
def generate_aggregated_recommendations(top_reasons_list, counts_str, summary_stats):
    try:
        input_data = {
            "top_reasons_list": ", ".join(top_reasons_list),
            "counts": counts_str,
            "summary_stats": summary_stats
        }

        result = _chain.invoke(input_data)

        return result.strip() if result else _fallback(top_reasons_list, counts_str, summary_stats)

    except Exception as e:
        logger.error(f"❌ Gemini error: {e}")
        return _fallback(top_reasons_list, counts_str, summary_stats)