import os
import json
import time
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq, APIConnectionError, AuthenticationError, RateLimitError, APIError
import uvicorn

# Load environment variables
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Sakhi - Women Safety AI",
    description="An empathetic and action-oriented AI companion for women's safety and support in India.",
    version="2.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --- Pydantic Models ---
class ChatPayload(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# =============================================================================
# 📚 RESOURCE DATABASE & MODEL CONFIG
# =============================================================================
LLM_MODEL = "llama3-70b-8192"

def load_resources():
    """Loads static resources for the chatbot."""
    resources = {
        "helplines": {
            "national_emergency": "112",
            "women_helpline": "181",
            "child_helpline": "1098",
            "cybercrime_helpline": "1930"
        },
        "legal_info": {
            "domestic_violence": "The Protection of Women from Domestic Violence Act, 2005 protects you from physical, emotional, and economic abuse. You have the right to a protection order.",
            "workplace_harassment": "The Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013 requires employers to form an Internal Complaints Committee (ICC)."
        },
        "ngos": {
            "mumbai": {"name": "Akshara Centre", "contact": "022-24316082"},
            "delhi": {"name": "Jagori", "contact": "011-26692700"},
            "bangalore": {"name": "Vimochana", "contact": "080-25492781"}
        },
        "self_care_tips": [
            "Take a few deep breaths. Inhale for 4 seconds, hold for 4, and exhale for 6.",
            "Find a quiet space if you can. Your safety and peace are important.",
            "Remember that your feelings are valid. It's okay to feel scared or upset."
        ]
    }
    return resources

RESOURCES = load_resources()

emergency_numbers = {
    "General Emergency Services": {
        "National Emergency Number": "112",
        "Police": "100",
        "Fire": "101",
        "Ambulance": "102",
        "Disaster Management Services": "108",
        "Cyber Crime Helpline": "1930",
        "Child Helpline (CHILDLINE)": "1098",
        "Senior Citizen Helpline": "14567",
        "Road Accident Emergency Service": "1073",
        "Railway Passenger Helpline": "139",
        "Mental Health Helpline": "080-46110007",
        "Blood Bank/Health Info": "104",
        "Anti Poison (New Delhi)": "1066 or 011-1066",
        "Railway Accident Emergency Service": "1072",
        "Relief Commissioner for Natural Calamities": "1070",
        "Central Vigilance Commission": "1964",
        "Tourist Helpline": "1363 or 1800111363",
        "LPG Leak Helpline": "1906",
        "Air Ambulance": "9540161344",
        "AIDS Helpline": "1097",
        "ORBO Centre, AIIMS (Organ Donation, Delhi)": "1060",
        "Call Centre": "1551"
    },
    "Women’s Safety Helplines": {
        "National Women Helpline": "181",
        "National Commission for Women Helpline": "7827170170",
        "Women Helpline (General)": "1091",
        "Women Power Line (Uttar Pradesh)": "1090",
        "Central Social Welfare Board - Police Helpline": ["1091", "1291", "(011) 23317004"],
        "Shakti Shalini": "10920",
        "Shakti Shalini - Women’s Shelter": ["(011) 24373736", "(011) 24373737"],
        "SAARTHAK": ["(011) 26853846", "(011) 26524061"],
        "All India Women’s Conference": ["10921", "(011) 23389680"],
        "Joint Women’s Programme (Branches in Bangalore, Kolkata, Chennai)": "(011) 24619821",
        "Sakshi - Violence Intervention Center": ["(0124) 2562336", "(0124) 5018873"],
        "Saheli - Women’s Organization": "(011) 24616485 (Available on Saturdays)",
        "Nirmal Niketan": "(011) 27859158",
        "Nari Raksha Samiti": "(011) 23973949",
        "RAHI (Recovering and Healing from Incest)": ["(011) 26238466", "(011) 26224042", "(011) 26227647"],
        "JAGORI": ["918800996640", "(011) 26692700"]
    }
}

# =============================================================================
# 🧠 MASTER SYSTEM PROMPTS
# =============================================================================
MASTER_SYSTEM_PROMPTS = {
    "DEFAULT": {
        "persona": """
        You are "Sakhi" – a trusted friend & protector AI for women’s safety in India.
        Core traits: empathy, authority, clarity, cultural awareness.
        Always speak in the user's language (Hindi, Hinglish, English, Marathi, etc.).
        Be short, natural, and supportive – like a strong but caring friend.
        """,
        "rules": """
        RESPONSE FORMAT:  
        🛡️ Support Message:  
        - Start with 1 empathetic short line.  
        - Then give 2–3 key safety/legal steps.  
        - End with a clear helpline list (📞 Important Helplines).  
        """
    },

    "EMERGENCY": {
        "persona": """
        You are "Sakhi", an urgent first-responder AI. 
        Tone: calm, direct, life-saving. 
        Only focus on immediate survival and safety.
        """,
        "rules": """
        RESPONSE FORMAT:  

        ⚠️ Emergency Help:  
        1. Call **112 immediately** (or 100 for Police).  
        2. Go to a safe place / nearest hospital.  
        3. Women Helpline: **1091** | NCW WhatsApp: **7827170170** - No small talk, no questions.  
        - Always reply in user’s language.  
        """
    },

    "LEGAL": {
        "persona": """
        You are "Sakhi", a legal rights guide for women. 
        Tone: empowering, concise, supportive.
        You explain rights in simple, short language.
        """,
        "rules": """
        RESPONSE FORMAT:  

        ⚖️ Legal Help:  
        - **FIR (First Information Report):** File FIR at nearest police station. Police cannot refuse.  
        - **Free Legal Aid:** Available via NALSA + NCW for women.  
        - **Protection Orders:** Court can grant restraining & compensation orders.  

        📞 Contacts:  
        - Women Helpline: **1091** - NCW Helpline (WhatsApp): **7827170170** - Emergency: **112** 👉 End with 1 empowering line: “Your rights are protected under law, you’re not alone.”  
        """
    },

    "CYBERCRIME": {
        "persona": """
        You are "Sakhi", a cybercrime protector AI. 
        Tone: practical, protective, reassuring.
        """,
        "rules": """
        RESPONSE FORMAT:  

        🖥️ Cybercrime Help:  
        - Helpline: **1930** - Secure accounts (change password, enable 2FA).  
        - Save evidence (screenshots, links).  
        - Report on **cybercrime.gov.in**.  

        👉 Reminder: “It’s not your fault, you are safe to report.”  
        """
    },

    "EMOTIONAL_SUPPORT": {
        "persona": """
        You are "Sakhi", an empathetic listener & safe space. 
        Tone: warm, gentle, like a caring friend.
        """,
        "rules": """
        RESPONSE FORMAT:  

        💜 Emotional Support:  
        - Start with 1 empathetic line (e.g., “I’m so sorry you’re going through this.”).  
        - Add 1 gentle question OR 1 calming tip.  
        - Keep it max 2 sentences.  
        """
    },
}




# =============================================================================
# 🤖 SAKHI CHATBOT CLASS
# =============================================================================
class SakhiChatbot:
    """The main class for the Sakhi Chatbot, managing state, intent, and responses."""
    def __init__(self, client: Groq):
        """Initializes the chatbot's state."""
        self.client = client
        self.chat_history: List[Dict] = []
        self.safety_status = "safe"  # Can be 'safe', 'unsafe', 'monitoring'
        self.user_location = None
        self.safe_circle = ["+919876543210", "+918765432109"] # Mock data

    def _call_groq_api(self, messages: list, temperature: float = 0.4, max_tokens: int = 150) -> str:
        """Helper function to call the Groq API with robust error handling."""
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except RateLimitError:
            return f"⚠️ I'm getting a lot of requests right now. Please wait a moment. For immediate help, call the National Emergency Helpline: {RESOURCES['helplines']['national_emergency']}."
        except APIError as e:
            print(f"API Error: {e}")
            return f"⚠️ My systems are facing a technical issue. For immediate help, please call the Women's Helpline: {RESOURCES['helplines']['women_helpline']}."
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return f"⚠️ I'm sorry, an unexpected error occurred. Please try again. If you need urgent assistance, call {RESOURCES['helplines']['national_emergency']}."

    def classify_intent(self, user_input: str) -> str:
        """Uses the LLM to classify the user's intent with high accuracy."""
        classification_prompt = f"""
        Analyze the user's message and classify its primary intent into ONE of the following categories:
        'EMERGENCY', 'LEGAL', 'CYBERCRIME', 'EMOTIONAL_SUPPORT', or 'GENERAL'.
        User's message: "{user_input}"
        Classification:
        """
        messages = [{"role": "user", "content": classification_prompt}]
        # Use a low-cost, fast model for classification if available, or the main one.
        response = self._call_groq_api(messages, temperature=0.0, max_tokens=20)
        intent = response.strip().upper().replace("'", "").replace('"',"")

        if intent in MASTER_SYSTEM_PROMPTS:
            return intent
        return "GENERAL"

    def _handle_special_commands(self, user_input: str) -> str | None:
        """Handles special slash commands for quick actions."""
        if user_input.lower().startswith("/location"):
            try:
                self.user_location = user_input.split(" ", 1)[1].strip()
                return f"Thank you. I've noted your location as {self.user_location}. This will help me provide more specific resources if you need them."
            except IndexError:
                return "Please provide a location after the command, like: /location Mumbai"
        
        if user_input.lower() == "/alert":
            return self.send_safe_circle_alert()
        return None

    def send_safe_circle_alert(self) -> str:
        """Simulates sending an alert to pre-configured trusted contacts."""
        print("\n[SYSTEM ACTION: Sending alert to Safe Circle...]")
        for number in self.safe_circle:
            print(f"  > SMS sent to {number}")
            time.sleep(0.1) # Simulate API call
        
        location_info = f"at location {self.user_location}" if self.user_location else "at their last known location"
        alert_message = f"Your Safe Circle has been alerted with the message: 'Emergency! Need help {location_info}.' Please also call {RESOURCES['helplines']['national_emergency']} immediately."
        self.safety_status = "unsafe"
        return alert_message

    def process_message(self, user_input: str) -> str:
        """Main function to generate a context-aware and safe response for the API."""
        command_response = self._handle_special_commands(user_input)
        if command_response:
            return command_response

        # --- MODIFIED SECTION ---
        # Keyword-based override for emergency intent
        emergency_keywords = ["help", "emergency", "danger", "attack", "unsafe", "assault"]
        user_input_lower = user_input.lower()
        if any(keyword in user_input_lower for keyword in emergency_keywords):
            intent = "EMERGENCY"
        else:
            intent = self.classify_intent(user_input)
        # --- END MODIFIED SECTION ---
        
        if intent == "EMERGENCY":
            self.safety_status = "unsafe"
        elif "safe" in user_input_lower or intent == "GENERAL":
            if self.safety_status == "unsafe":
                self.safety_status = "monitoring"
        
        prompt_data = MASTER_SYSTEM_PROMPTS.get(intent, MASTER_SYSTEM_PROMPTS["DEFAULT"])
        
        contextual_info = f"""
        CURRENT CONTEXT:
        - User's Safety Status: {self.safety_status}
        - User's Location: {self.user_location or 'Not Provided'}
        - Available Helplines: {json.dumps(RESOURCES['helplines'])}
        - All Emergency Numbers: {json.dumps(emergency_numbers)}
        - Available NGOs: {json.dumps(RESOURCES['ngos'])}
        - Available Legal Info: {json.dumps(RESOURCES['legal_info'])}
        """
        
        # Add specific instruction to prevent question repetition
        anti_repetition_rule = """
        CRITICAL: Do NOT repeat or translate the user's question. Answer directly without echoing their words.
        """
        
        full_system_prompt = f"{prompt_data['persona']}\n{contextual_info}\nRULES:\n{prompt_data['rules']}\n{anti_repetition_rule}"

        messages = [
            {"role": "system", "content": full_system_prompt},
            *self.chat_history[-6:],
            {"role": "user", "content": user_input}
        ]

        response_text = self._call_groq_api(messages)
        
        self.chat_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response_text}
        ])
        
        return response_text

# --- Initialize the Assistant ---
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in a .env file.")

    client = Groq(api_key=groq_api_key)
    client.models.list()
    print("Groq API key successfully validated.")

    # Create a global instance of the chatbot, passing the client to it
    assistant = SakhiChatbot(client=client)
    print("🌸 Sakhi - Your Safety Companion is ready. 🌸")

except (ValueError, AuthenticationError, APIConnectionError, APIError) as e:
    print(f"\nFatal Initialization Error: {e}")
    print("Sakhi cannot start. Please ensure your Groq API key is correctly set.")
    assistant = None
except Exception as e:
    print(f"\nCritical startup error: {type(e).__name__} - {e}")
    print("Sakhi cannot start due to an unforeseen issue.")
    assistant = None

# --- API Endpoint for Chatting ---
@app.post("/chat")
async def chat(payload: ChatPayload):
    """
    Handle chat requests from the frontend.
    Receives a message, processes it with the AI assistant, and returns a reply.
    """
    if not assistant:
        raise HTTPException(status_code=500, detail="Chatbot is not initialized. Please check the server logs.")

    user_input = payload.message
    if not user_input.strip():
        return ChatResponse(reply="Please say something.")

    try:
        response = assistant.process_message(user_input)
        # FIX: Ensure proper UTF-8 encoding for Hindi text
        return ChatResponse(reply=response)
    except Exception as e:
        print(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your message.")

# --- Server Startup ---
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=5000)
