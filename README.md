**AI Resume Screening and Candidate Ranking System**

**Project Overview:**
The AI Resume Screening and Candidate Ranking System is an end-to-end analytics and AI application that screens resumes, matches candidates to job descriptions, ranks applicants, and generates explainable, bias-aware hiring insights.

This project simulates how modern recruiting, people analytics, and product teams use AI-assisted tools to improve hiring efficiency while maintaining transparency and human oversight.

**Project Goal:**

-Build an AI-powered system that helps recruiters and hiring managers:

-Quickly evaluate large volumes of resumes

-Understand why candidates are ranked a certain way

-Identify skill gaps and strengths

-Detect potential bias signals before making hiring decisions

**Core Problem**

-Time-consuming

-Inconsistent across reviewers

-Difficult to explain or audit

-Vulnerable to unconscious bias

-Most automated screening tools act as black boxes, providing scores without clarity into how decisions are made.

**Why I Built This:**
I wanted to build a project that reflects how real analytics and product teams approach responsible AI in high-stakes domains like hiring.

My goal was to go beyond simple resume matching by:

Combining AI with rule-based analytics

Prioritizing explainability over opaque scoring

Introducing bias-awareness instead of claiming “bias-free AI”

This project focuses on decision support, not decision replacement.


**What This System Does**

-Paste a job description

-Upload multiple resumes (PDF, DOCX, or TXT)

-Automatically extract job-relevant skills

-Rank candidates using a hybrid scoring framework

-Review matched and missing skills per candidate

-See evidence snippets from resumes

-Generate recruiter-friendly explanations

-Flag cases where sensitive information may influence rankings

-Export ranked results as a CSV for reporting or review


**Architecture Overview**
Job Description-->Skill and Requirement Extraction--> Resume Parsing (PDF / DOCX / TXT)--> Semantic Embeddings + Rule-Based Analysis--> Hybrid Candidate Scoring--> Bias Detection & Score Comparison--> Explainable Results & Insights--> CSV Export

**Languages & Tools**

-Python

-Streamlit

-Pandas

-NumPy

-AI & Analytics

-OpenAI API (Embeddings + LLM explanations)

-VS Code

Integrate AI responsibly with transparency and oversight

Design products that balance automation with human decision-making

**Demo Video:** (used a job description: https://bmo.wd3.myworkdayjobs.com/en-US/External/details/Senior-Manager--Behavioral-Analytics-Model-Validation_R260001273?q=business+analyst for this demo)

https://github.com/user-attachments/assets/dc21b87d-5234-4339-bb13-32d99c154fcb

