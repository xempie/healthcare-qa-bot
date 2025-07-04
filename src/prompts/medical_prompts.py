"""
Medical prompt templates for the Healthcare Q&A Bot.
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# System prompt for medical Q&A
MEDICAL_QA_SYSTEM_PROMPT = """
You are a helpful healthcare information assistant. Your role is to provide accurate, evidence-based health information based on the medical documents provided to you.

IMPORTANT GUIDELINES:
1. Always provide accurate information based on the provided medical documents
2. Never provide specific medical advice, diagnoses, or treatment recommendations
3. Always encourage users to consult with healthcare professionals for personal medical concerns
4. If you don't have information about a specific topic, clearly state that
5. Use clear, simple language that patients can understand
6. Include relevant medical disclaimers when appropriate

RESPONSE FORMAT:
- Provide clear, concise answers
- Use bullet points for lists when helpful
- Include source references when possible
- End with appropriate medical disclaimers

Remember: You are providing general health information, not personalized medical advice.
"""

# Human prompt template for Q&A
MEDICAL_QA_HUMAN_PROMPT = """
Context from medical documents:
{context}

Patient Question: {question}

Please provide a helpful response based on the medical documents provided. Include relevant information and appropriate medical disclaimers.
"""

# Chat prompt template
MEDICAL_QA_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(MEDICAL_QA_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(MEDICAL_QA_HUMAN_PROMPT)
])

# Retrieval prompt for document search
RETRIEVAL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
Based on the following patient question, generate search queries to find relevant medical information:

Patient Question: {question}

Generate 2-3 relevant search queries that would help find medical information to answer this question:
"""
)

# Safety classification prompt
SAFETY_CLASSIFICATION_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
Classify the following patient question for safety and appropriateness:

Question: {question}

Classify this question as:
1. SAFE - General health information question that can be answered with educational content
2. EMERGENCY - Requires immediate medical attention
3. RESTRICTED - Requires specific medical advice that should only come from healthcare professionals
4. INAPPROPRIATE - Not suitable for a healthcare Q&A system

Provide your classification and a brief explanation:
Classification: 
Explanation: 
"""
)

# Follow-up questions prompt
FOLLOW_UP_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
Based on the patient's question and the provided answer, suggest 2-3 relevant follow-up questions that the patient might want to ask:

Original Question: {question}
Answer Provided: {answer}

Suggested follow-up questions:
1. 
2. 
3. 
"""
)

# Conversation summary prompt
CONVERSATION_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["conversation_history"],
    template="""
Summarize the following conversation between a patient and healthcare information assistant:

Conversation History:
{conversation_history}

Provide a concise summary of:
1. Main topics discussed
2. Key information provided
3. Any recommendations made
4. Follow-up actions suggested

Summary:
"""
)

# Document relevance scoring prompt
DOCUMENT_RELEVANCE_PROMPT = PromptTemplate(
    input_variables=["question", "document_content"],
    template="""
Score the relevance of this document content to the patient's question on a scale of 1-10:

Patient Question: {question}

Document Content: {document_content}

Relevance Score (1-10): 
Explanation: 
"""
)

# Medical terminology explanation prompt
TERMINOLOGY_EXPLANATION_PROMPT = PromptTemplate(
    input_variables=["term", "context"],
    template="""
Explain the following medical term in simple, patient-friendly language:

Medical Term: {term}
Context: {context}

Explanation:
- Definition in simple terms
- Why it's important
- How it relates to the patient's question
"""
)

# Symptom checker disclaimer prompt
SYMPTOM_CHECKER_DISCLAIMER = """
**Important:** This information is for educational purposes only and cannot replace professional medical evaluation. 

If you are experiencing symptoms:
- Consult with a healthcare provider for proper evaluation
- Seek immediate medical attention for severe or emergency symptoms
- Do not delay professional medical care based on this information

This tool provides general health information and should not be used for self-diagnosis or treatment decisions.
"""

# Medication information disclaimer
MEDICATION_DISCLAIMER = """
**Medication Information Disclaimer:**
- This information is for educational purposes only
- Never start, stop, or change medications without consulting your healthcare provider
- Always follow your doctor's instructions and prescription labels
- Discuss any concerns about medications with your pharmacist or doctor
- Report any side effects to your healthcare provider immediately
"""
