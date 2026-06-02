import streamlit as st
import json
import re
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from pydantic import BaseModel, Field
from typing import List

# 1. Define the Structured Output Schema
class ParameterDetail(BaseModel):
    parameter_name: str = Field(..., description="The name of the metric (e.g., Expense Ratio, Sharpe Ratio)")
    value: str = Field(..., description="The calculated or current value with units")
    indicator: str = Field(..., description="Visual indicator emoji: 🟢, 🟡, or 🔴")
    justification: str = Field(..., description="Short explanation of why this indicator was chosen")

class MutualFundReport(BaseModel):
    fund_name: str
    overall_summary: str
    parameters: List[ParameterDetail] = Field(..., min_items=10, max_items=10)

# Helper function to clean and parse JSON robustly
def safe_parse_json(content):
    if not content:
        raise ValueError("The agent returned an empty response.")
        
    # If it's already a Pydantic object, return it directly
    if isinstance(content, MutualFundReport):
        return content
    
    # If it's a dictionary, unpack it
    if isinstance(content, dict):
        return MutualFundReport(**content)
        
    if isinstance(content, str):
        # Remove potential Markdown JSON wrapper formatting if present
        cleaned_content = re.sub(r"^```json\s*", "", content, flags=re.MULTILINE)
        cleaned_content = re.sub(r"\s*```$", "", cleaned_content, flags=re.MULTILINE)
        cleaned_content = cleaned_content.strip()
        
        # Parse and return
        data_dict = json.loads(cleaned_content)
        return MutualFundReport(**data_dict)
        
    raise ValueError(f"Unexpected response format: {type(content)}")

# Streamlit UI Configuration
st.set_page_config(page_title="Agentic Mutual Fund Analyzer", layout="wide")
st.title("📊 Free Agentic Mutual Fund Report Generator")
st.caption("Powered by Agno, Gemini Flash, and Streamlit")

with st.sidebar:
    st.header("Setup")
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("[Get a free Gemini API Key here](https://aistudio.google.com/)")

fund_input = st.text_input("Enter Mutual Fund Name (e.g., Vanguard 500 Index Fund)", "")

if st.button("Generate Report") and fund_input:
    if not gemini_key:
        st.error("Please provide a Gemini API Key in the sidebar.")
    else:
        with st.spinner("Agentic researchers are analyzing the fund details..."):
            try:
                # 2. Define Agent WITH response_model configured at initialization
                analyzer_agent = Agent(
                    model=Gemini(id="gemini-2.5-flash", api_key=gemini_key),
                    tools=[YFinanceTools(), DuckDuckGoTools()],
                    instructions=[
                        "You are a Senior Mutual Fund Research Analyst.",
                        "Search web data and Yahoo Finance to fetch actual data for the requested fund.",
                        "You MUST strictly output a valid JSON matching the schema.",
                        "Provide exactly 10 financial metrics: 1. Net Asset Value (NAV), 2. Expense Ratio, "
                        "3. Assets Under Management (AUM), 4. Sharpe Ratio, 5. Alpha, 6. Beta, "
                        "7. 3-Year Return CAGR, 8. 5-Year Return CAGR, 9. Portfolio Turnover Rate, 10. Standard Deviation."
                    ],
                    response_model=MutualFundReport
                )

                # 3. Fire Agent Run
                response = analyzer_agent.run(f"Analyze the following mutual fund: {fund_input}")
                
                # Use our robust parsing function
                report_data = safe_parse_json(response.content)

                # 4. Render the UI Cleanly
                st.header(f"📈 Performance Report: {report_data.fund_name}")
                st.subheader("Executive Summary")
                st.write(report_data.overall_summary)
                
                st.write("---")
                st.subheader("Metric Evaluation Matrix")

                for item in report_data.parameters:
                    with st.container():
                        col1, col2, col3 = st.columns([1, 3, 6])
                        col1.markdown(f"### {item.indicator}")
                        col2.markdown(f"**{item.parameter_name}**\n\n`{item.value}`")
                        col3.markdown(f"*Analysis:* {item.justification}")
                        st.markdown("---")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                # Provide debug info in an expander for troubleshooting
                if 'response' in locals() and hasattr(response, 'content'):
                    with st.expander("View Raw Agent Output"):
                        st.code(response.content)
