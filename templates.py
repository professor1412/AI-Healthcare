from pydantic import BaseModel, Field
from typing import Literal
from langchain.prompts import PromptTemplate


cardiologist_template = """
You are a cardiologist tasked with reviewing a patient's medical report. 
Analyze ECG, blood test results, Holter monitor data, and echocardiogram. 
Determine potential cardiac concerns such as arrhythmias or structural abnormalities. 
Provide possible diagnoses, recommended follow-up tests, and treatment suggestions.
Include a disclaimer regarding the limitations of AI-generated medical advice.
"""

psychologist_template = """
You are a clinical psychologist. Review the patient's report for signs of anxiety, 
depression, trauma, or other psychological issues. 
Offer a concise mental health evaluation and therapeutic suggestions.
Include a disclaimer noting limitations of AI in diagnosing or replacing human consultation.
"""

pulmonologist_template = """
You are a pulmonologist. Review the patient's respiratory-related data including 
symptoms, medical history, and diagnostic test results. 
Identify possible conditions like asthma, COPD, or infections. 
Recommend tests or treatments based on the evaluation. 
Include a disclaimer regarding the AI limitations in medical contexts.
"""

team_template = """
---
### Multidisciplinary Medical Report
**Task Overview:**
You are a multidisciplinary team consisting of a Cardiologist, Psychologist, and Pulmonologist.

**Goal:**
Review and integrate findings from each specialist to produce a holistic medical report:
Cardiologist Report: {cardiologist_report}
Psychologist Report: {psychologist_report}
Pulmonologist Report: {pulmonologist_report}

**Report Structure:**
1. **Summary of Findings:**
   - Key issues identified by each specialist.
2. **Integrated Analysis:**
   - Cross-specialty interpretation of symptoms.
3. **Recommendations:**
   - Unified next steps including further tests, treatments, or referrals.
4. **Disclaimer:**
   - Acknowledge the AI-generated nature of the report and recommend human consultation.
---
"""

# Create PromptTemplate using LangChain
multidisciplinary_prompt = PromptTemplate.from_template(template=team_template)

# Pydantic models for the responses
class MedicalReportResponse(BaseModel):
    """Respond with a comprehensive AI-generated medical report."""
    return_direct: bool = False
    summary_of_findings: str = Field(description="Summary of health issues identified by all specialists")
    integrated_analysis: str = Field(description="Cross-specialty analysis of the patient’s symptoms")
    recommendations: str = Field(description="Unified set of next steps for the patient")
    disclaimer: str = Field(description="AI-generated report disclaimer for medical limitations")

class CardiologistResponse(BaseModel):
    return_direct: bool = False
    possible_diagnoses: str = Field(description="Possible cardiac conditions based on the report")
    suggested_tests: str = Field(description="Follow-up tests or monitoring recommendations")
    treatment_advice: str = Field(description="Suggested treatments or referrals")
    disclaimer: str = Field(description="AI-generated cardiac advice disclaimer")

class PsychologistResponse(BaseModel):
    return_direct: bool = False
    mental_health_findings: str = Field(description="Signs of psychological issues")
    therapy_suggestions: str = Field(description="Suggestions for mental health support")
    disclaimer: str = Field(description="AI-generated psychological advice disclaimer")

class PulmonologistResponse(BaseModel):
    return_direct: bool = False
    respiratory_issues: str = Field(description="Likely pulmonary issues or conditions")
    testing_recommendations: str = Field(description="Recommended imaging or pulmonary tests")
    treatment_options: str = Field(description="Treatment plan suggestions")
    disclaimer: str = Field(description="AI-generated pulmonology advice disclaimer")
