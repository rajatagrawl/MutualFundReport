import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from pydantic import BaseModel, Field
from typing import List

# 1. Define the Structured Output Schema for 10 Parameters
class ParameterDetail(BaseModel):
    parameter_name: str = Field(..., description="The name of the metric (e.g., Expense Ratio, Sharpe Ratio)")
    value: str = Field(..., description="The calculated or current value with units")
    indicator: str = Field(..., description="Visual indicator emoji: 🟢 (Strong/Good), 🟡 (Neutral), or 🔴 (Weak/High Risk) based on industry benchmarks")
    justification: str = Field(..., description="Short explanation of why this indicator was chosen")

class MutualFundReport(BaseModel):
    fund_name: str
    overall_summary: str
    parameters: List[ParameterDetail] = Field(..., min_items=10, max_items=10)

# Streamlit UI Configuration
st.set_page_config(page_title="Agentic Mutual Fund Analyzer", layout="wide")
st.title("📊 Free Agentic Mutual Fund Report Generator")
st.caption("Powered by Agno, Gemini Flash, and Streamlit")

# Sidebar for API Key Setup (Keeps it secure and free)
with st.sidebar:
    st.header("Setup")
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("[Get a free Gemini API Key here](https://aistudio.google.com/)")

fund_input = st.text_input("Enter Mutual Fund Name (e.g., Vanguard 500 Index Fund or Parag Parikh Flexi Cap)", "")

if st.button("Generate Report") and fund_input:
    if not gemini_key:
        st.error("Please provide a Gemini API Key in the sidebar.")
    else:
        with st.spinner("Agentic researchers are analyzing the fund details..."):
            try:
                # 2. Configure the Agent with Web Tools and Gemini
                analyzer_agent = Agent(
                    model=Gemini(id="gemini-2.5-flash", api_key=gemini_key),
                    tools=[YFinanceTools(include_tools=["get_current_stock_price"]), DuckDuckGoTools(include_tools=["web_search"])], 
                    # Instead of tools=[YFinanceTools(), DuckDuckGoTools()]                  
                    instructions=[
                        "You are a Senior Mutual Fund Research Analyst.",
                        "Search web data and Yahoo Finance to fetch data for the requested fund.",
                        "Provide a strict structured output mapping EXACTLY 10 financial metrics:",
                        "1. Net Asset Value (NAV), 2. Expense Ratio, 3. Assets Under Management (AUM), "
                        "4. Sharpe Ratio, 5. Alpha, 6. Beta, 7. 3-Year Return CAGR, 8. 5-Year Return CAGR, "
                        "9. Portfolio Turnover Rate, 10. Standard Deviation.",
                        "Evaluate each parameter against typical financial benchmarks to assign a 🟢, 🟡, or 🔴 indicator."
                    ],
                    #response_model=MutualFundReport, # Forces structured JSON matching our Pydantic schema
                    markdown=False,
                    add_history_to_context=False
                )


                # 3. Pass the response_model directly into the run method
                # (Check if your version of Agno supports this override)
                response = analyzer_agent.run(
                    f"Analyze the following mutual fund: {fund_input}",
                    response_model=MutualFundReport 
                )

                # 3. Fire the Agent Run
                # response = analyzer_agent.run(f"Analyze the following mutual fund: {fund_input}")
                report_data = response.content

                # SAFEST GUARD: If Agno returned a string representation of the JSON instead of the object
                # if isinstance(report_data, str):
                #     import json
                #     # Parse string to dict, then unpack into your Pydantic model
                #     data_dict = json.loads(report_data)
                #     report_data = MutualFundReport(**data_dict)


                # REPLACES YOUR OLD SAFEST GUARD BLOCK
                if isinstance(report_data, str):
                    import json
                    
                    # Strip away accidental markdown code block wrappers if present
                    cleaned_data = report_data.strip()
                    if cleaned_data.startswith("```json"):
                        cleaned_data = cleaned_data.split("```json")[1].split("```")[0].strip()
                    elif cleaned_data.startswith("```"):
                        cleaned_data = cleaned_data.split("```")[1].split("```")[0].strip()
                        
                    data_dict = json.loads(cleaned_data)
                    report_data = MutualFundReport(**data_dict)

              
                # 4. Render the UI cleanly from Structured Data
                st.header(f"📈 Performance Report: {report_data.fund_name}")
                st.subheader("Executive Summary")
                st.write(report_data.overall_summary)
                
                st.write("---")
                st.subheader("Metric Evaluation Matrix")

                # Displaying metrics inside neat Streamlit columns
                for item in report_data.parameters:
                    with st.container():
                        col1, col2, col3 = st.columns([1, 3, 6])
                        col1.markdown(f"### {item.indicator}")
                        col2.markdown(f"**{item.parameter_name}**\n\n`{item.value}`")
                        col3.markdown(f"*Analysis:* {item.justification}")
                        st.markdown("---")

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "ResourceExhausted" in error_msg:
                    st.error("🛑 **Gemini API Rate Limit Hit (Error 429):** You've made too many requests too quickly. Please wait about 20–30 seconds and try clicking 'Generate Report' again.")
                else:
                    st.error(f"An unexpected error occurred: {error_msg}")
