"""
Validation utilities for healthcare Q&A bot.
"""

import re
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config.settings import EMERGENCY_KEYWORDS, RESTRICTED_KEYWORDS


def validate_medical_query(query: str) -> Dict[str, any]:
    """Validate a medical query for safety and appropriateness."""
    
    if not query or not isinstance(query, str):
        return {
            "is_valid": False,
            "reason": "Empty or invalid query",
            "severity": "low"
        }
    
    query = query.strip()
    
    if len(query) < 3:
        return {
            "is_valid": False,
            "reason": "Query too short",
            "severity": "low"
        }
    
    if len(query) > 1000:
        return {
            "is_valid": False,
            "reason": "Query too long (max 1000 characters)",
            "severity": "medium"
        }
    
    # Check for emergency situations
    emergency_check = is_emergency_query(query)
    if emergency_check["is_emergency"]:
        return {
            "is_valid": True,
            "is_emergency": True,
            "emergency_type": emergency_check["emergency_type"],
            "severity": "critical"
        }
    
    # Check for restricted content
    restricted_check = is_restricted_query(query)
    if restricted_check["is_restricted"]:
        return {
            "is_valid": True,
            "is_restricted": True,
            "restriction_type": restricted_check["restriction_type"],
            "severity": "high"
        }
    
    # Check for inappropriate content
    inappropriate_check = is_inappropriate_query(query)
    if inappropriate_check["is_inappropriate"]:
        return {
            "is_valid": False,
            "reason": inappropriate_check["reason"],
            "severity": "medium"
        }
    
    return {
        "is_valid": True,
        "severity": "low"
    }


def is_emergency_query(query: str) -> Dict[str, any]:
    """Check if query indicates a medical emergency."""
    
    query_lower = query.lower()
    
    # Emergency keywords with severity levels
    critical_keywords = [
        "heart attack", "stroke", "can't breathe", "cannot breathe",
        "unconscious", "not breathing", "severe chest pain",
        "heavy bleeding", "overdose", "poisoning"
    ]
    
    urgent_keywords = [
        "emergency", "urgent", "chest pain", "difficulty breathing",
        "severe pain", "bleeding heavily", "suicide", "self-harm"
    ]
    
    # Check for critical emergencies
    for keyword in critical_keywords:
        if keyword in query_lower:
            return {
                "is_emergency": True,
                "emergency_type": "critical",
                "detected_keyword": keyword,
                "confidence": "high"
            }
    
    # Check for urgent situations
    for keyword in urgent_keywords:
        if keyword in query_lower:
            return {
                "is_emergency": True,
                "emergency_type": "urgent",
                "detected_keyword": keyword,
                "confidence": "medium"
            }
    
    # Pattern-based emergency detection
    emergency_patterns = [
        r"call\s+(911|ambulance|emergency)",
        r"(having|experiencing)\s+.*(heart attack|stroke)",
        r"can'?t\s+(breath|breathe)",
        r"severe\s+(pain|bleeding)",
        r"think\s+.*(dying|heart attack)"
    ]
    
    for pattern in emergency_patterns:
        if re.search(pattern, query_lower):
            return {
                "is_emergency": True,
                "emergency_type": "pattern_detected",
                "detected_pattern": pattern,
                "confidence": "medium"
            }
    
    return {"is_emergency": False}


def is_restricted_query(query: str) -> Dict[str, any]:
    """Check if query requires professional medical consultation."""
    
    query_lower = query.lower()
    
    # Diagnosis-related restrictions
    diagnosis_keywords = [
        "diagnose", "diagnosis", "what do i have", "do i have",
        "is this", "could this be", "am i having"
    ]
    
    # Treatment/medication restrictions
    treatment_keywords = [
        "should i take", "how much", "dosage", "prescription",
        "medicine for", "treatment for", "cure for"
    ]
    
    # Medical advice restrictions
    advice_keywords = [
        "should i", "medical advice", "what should i do",
        "recommend", "suggest treatment", "best treatment"
    ]
    
    # Check for diagnosis requests
    for keyword in diagnosis_keywords:
        if keyword in query_lower:
            return {
                "is_restricted": True,
                "restriction_type": "diagnosis_request",
                "detected_keyword": keyword,
                "reason": "Diagnosis requests require professional medical evaluation"
            }
    
    # Check for treatment/medication requests
    for keyword in treatment_keywords:
        if keyword in query_lower:
            return {
                "is_restricted": True,
                "restriction_type": "treatment_advice",
                "detected_keyword": keyword,
                "reason": "Treatment and medication advice must come from healthcare professionals"
            }
    
    # Check for medical advice requests
    for keyword in advice_keywords:
        if keyword in query_lower:
            return {
                "is_restricted": True,
                "restriction_type": "medical_advice",
                "detected_keyword": keyword,
                "reason": "Specific medical advice requires professional consultation"
            }
    
    # Pattern-based restrictions
    restricted_patterns = [
        r"(should|can)\s+i\s+(take|use|stop)",
        r"how\s+(much|many)\s+.*(pill|medication|drug)",
        r"is\s+it\s+(safe|okay)\s+to",
        r"(replace|substitute)\s+.*(medication|prescription)",
        r"(stop|start|change)\s+.*(medication|treatment)"
    ]
    
    for pattern in restricted_patterns:
        if re.search(pattern, query_lower):
            return {
                "is_restricted": True,
                "restriction_type": "medication_safety",
                "detected_pattern": pattern,
                "reason": "Medication safety questions require professional guidance"
            }
    
    return {"is_restricted": False}


def is_inappropriate_query(query: str) -> Dict[str, any]:
    """Check if query is inappropriate for healthcare Q&A."""
    
    query_lower = query.lower()
    
    # Non-medical topics
    non_medical_keywords = [
        "weather", "sports", "politics", "entertainment", "cooking",
        "travel", "technology", "business", "finance", "legal advice"
    ]
    
    # Inappropriate content
    inappropriate_keywords = [
        "illegal", "drug abuse", "recreational drugs", "getting high",
        "fake prescription", "drug dealer", "black market"
    ]
    
    # Check for non-medical content
    for keyword in non_medical_keywords:
        if keyword in query_lower and not any(medical in query_lower for medical in ["health", "medical", "doctor", "symptom"]):
            return {
                "is_inappropriate": True,
                "reason": f"Query appears to be about {keyword}, not healthcare",
                "category": "non_medical"
            }
    
    # Check for inappropriate content
    for keyword in inappropriate_keywords:
        if keyword in query_lower:
            return {
                "is_inappropriate": True,
                "reason": "Query contains inappropriate content for healthcare assistance",
                "category": "inappropriate_content"
            }
    
    # Check for nonsensical or spam-like content
    if is_nonsensical_query(query):
        return {
            "is_inappropriate": True,
            "reason": "Query appears to be nonsensical or spam",
            "category": "spam_or_nonsense"
        }
    
    return {"is_inappropriate": False}


def is_nonsensical_query(query: str) -> bool:
    """Check if query is nonsensical or spam-like."""
    
    # Check for repeated characters
    if re.search(r'(.)\1{5,}', query):
        return True
    
    # Check for random character sequences
    if re.search(r'[a-zA-Z]{20,}', query) and not any(word in query.lower() for word in ["symptom", "condition", "treatment", "health"]):
        return True
    
    # Check word-to-character ratio
    words = query.split()
    if len(words) > 3:
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 2 or avg_word_length > 15:
            return True
    
    return False


def validate_patient_age(age_input: str) -> Dict[str, any]:
    """Validate patient age input."""
    
    if not age_input:
        return {
            "is_valid": False,
            "reason": "Age not provided"
        }
    
    try:
        age = int(age_input)
        
        if age < 0:
            return {
                "is_valid": False,
                "reason": "Age cannot be negative"
            }
        
        if age > 150:
            return {
                "is_valid": False,
                "reason": "Age seems unrealistic"
            }
        
        # Age-specific considerations
        age_category = categorize_age(age)
        
        return {
            "is_valid": True,
            "age": age,
            "age_category": age_category,
            "special_considerations": get_age_considerations(age_category)
        }
        
    except ValueError:
        return {
            "is_valid": False,
            "reason": "Age must be a valid number"
        }


def categorize_age(age: int) -> str:
    """Categorize age into medical age groups."""
    
    if age < 1:
        return "infant"
    elif age < 13:
        return "child"
    elif age < 18:
        return "adolescent"
    elif age < 65:
        return "adult"
    else:
        return "senior"


def get_age_considerations(age_category: str) -> List[str]:
    """Get special considerations for different age groups."""
    
    considerations = {
        "infant": [
            "Pediatric dosing required",
            "Immediate pediatrician consultation recommended",
            "Special safety considerations for infants"
        ],
        "child": [
            "Pediatric consultation recommended",
            "Age-appropriate explanations needed",
            "Parental involvement required"
        ],
        "adolescent": [
            "Age-appropriate health education",
            "Privacy considerations",
            "Growth and development factors"
        ],
        "adult": [
            "Standard adult health information applicable"
        ],
        "senior": [
            "Multiple medication considerations",
            "Age-related health changes",
            "Fall risk awareness"
        ]
    }
    
    return considerations.get(age_category, [])


def validate_symptom_input(symptoms: str) -> Dict[str, any]:
    """Validate symptom description input."""
    
    if not symptoms or not symptoms.strip():
        return {
            "is_valid": False,
            "reason": "No symptoms provided"
        }
    
    symptoms = symptoms.strip()
    
    if len(symptoms) < 5:
        return {
            "is_valid": False,
            "reason": "Symptom description too brief"
        }
    
    if len(symptoms) > 2000:
        return {
            "is_valid": False,
            "reason": "Symptom description too long"
        }
    
    # Check for emergency symptoms
    emergency_check = is_emergency_query(symptoms)
    if emergency_check["is_emergency"]:
        return {
            "is_valid": True,
            "is_emergency": True,
            "emergency_details": emergency_check,
            "immediate_action_required": True
        }
    
    # Extract and validate individual symptoms
    symptom_list = extract_symptoms(symptoms)
    
    return {
        "is_valid": True,
        "symptoms": symptom_list,
        "symptom_count": len(symptom_list),
        "requires_medical_attention": assess_symptom_severity(symptom_list)
    }


def extract_symptoms(symptom_text: str) -> List[str]:
    """Extract individual symptoms from text."""
    
    # Common symptom keywords
    symptom_keywords = [
        "pain", "ache", "fever", "headache", "nausea", "vomiting",
        "diarrhea", "constipation", "fatigue", "weakness", "dizziness",
        "shortness of breath", "cough", "sore throat", "runny nose",
        "rash", "swelling", "bruising", "bleeding", "tingling",
        "numbness", "burning", "itching", "cramping"
    ]
    
    found_symptoms = []
    text_lower = symptom_text.lower()
    
    for symptom in symptom_keywords:
        if symptom in text_lower:
            found_symptoms.append(symptom)
    
    # Also split by common separators
    potential_symptoms = re.split(r'[,;.]|\sand\s|\sor\s', symptom_text)
    
    for symptom in potential_symptoms:
        symptom = symptom.strip()
        if len(symptom) > 3 and symptom.lower() not in found_symptoms:
            found_symptoms.append(symptom.lower())
    
    return found_symptoms


def assess_symptom_severity(symptoms: List[str]) -> str:
    """Assess the severity level of symptoms."""
    
    high_severity_indicators = [
        "severe", "intense", "unbearable", "crushing", "sharp",
        "sudden", "worsening", "getting worse"
    ]
    
    moderate_severity_indicators = [
        "moderate", "persistent", "ongoing", "daily", "frequent"
    ]
    
    symptoms_text = " ".join(symptoms).lower()
    
    if any(indicator in symptoms_text for indicator in high_severity_indicators):
        return "high"
    elif any(indicator in symptoms_text for indicator in moderate_severity_indicators):
        return "moderate"
    else:
        return "low"


def validate_medication_query(query: str) -> Dict[str, any]:
    """Validate medication-related queries."""
    
    medication_indicators = [
        "medication", "medicine", "drug", "pill", "tablet",
        "capsule", "prescription", "dosage", "dose"
    ]
    
    query_lower = query.lower()
    
    is_medication_query = any(indicator in query_lower for indicator in medication_indicators)
    
    if not is_medication_query:
        return {
            "is_medication_query": False
        }
    
    # Check for dangerous medication queries
    dangerous_patterns = [
        r"overdose",
        r"too much",
        r"maximum dose",
        r"double dose",
        r"mix.*alcohol",
        r"safe.*pregnancy"
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query_lower):
            return {
                "is_medication_query": True,
                "is_dangerous": True,
                "requires_immediate_consultation": True,
                "detected_concern": pattern
            }
    
    return {
        "is_medication_query": True,
        "is_dangerous": False,
        "requires_consultation": True,
        "safety_disclaimer_required": True
    }


def sanitize_input(text: str) -> str:
    """Sanitize user input for safety."""
    
    if not text:
        return ""
    
    # Remove potentially harmful characters
    text = re.sub(r'[<>\"\'&]', '', text)
    
    # Limit length
    if len(text) > 2000:
        text = text[:2000]
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def format_validation_response(validation_result: Dict[str, any]) -> str:
    """Format validation results into user-friendly messages."""
    
    if not validation_result.get("is_valid", True):
        reason = validation_result.get("reason", "Invalid query")
        return f"I'm sorry, but your query couldn't be processed: {reason}. Please try rephrasing your question."
    
    if validation_result.get("is_emergency"):
        return "This appears to be a medical emergency. Please call 911 or your local emergency services immediately. Do not rely on online information for emergency situations."
    
    if validation_result.get("is_restricted"):
        reason = validation_result.get("reason", "This query requires professional consultation")
        return f"{reason}. Please consult with a healthcare provider for personalized medical advice."
    
    return "Query validated successfully."
