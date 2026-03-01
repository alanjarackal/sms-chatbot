import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient
from datetime import datetime, timedelta
import json
import pytz

# Google Calendar API imports
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nyaya Sahayi Backend")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Gemini Configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Using the fast, efficient model for routing and extraction
model = genai.GenerativeModel('gemini-2.5-flash-lite') 

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
if MONGO_URI:
    db = MongoClient(MONGO_URI)["nyaya_db"]
    clients_collection = db["clients"]
    appointments_collection = db["appointments"]
else:
    logger.warning("MONGO_URI not set. Database features will fail.")
    clients_collection = None
    appointments_collection = None

# --- DATA MODELS ---
class HistoryItem(BaseModel):
    role: str  # "user" or "ai"
    content: str

class ChatRequest(BaseModel):
    message: str
    client_id: str = "default_client"
    history: List[HistoryItem] = []  # Added memory

# --- HELPER: FORMAT HISTORY ---
def format_history(history: List[HistoryItem]) -> str:
    """Converts the history array into a readable text block for the AI."""
    if not history:
        return "No previous conversation."
    # Keep the last 6 interactions (12 messages) to save tokens
    return "\n".join([f"{h.role.upper()}: {h.content}" for h in history[-12:]]) 

# ==========================================
# GOOGLE CALENDAR INTEGRATION
# ==========================================
def get_available_slots() -> list:
    """Fetches real available slots from the lawyer's Google Calendar."""
    if not os.path.exists("service_account.json"):
        logger.warning("service_account.json not found. Using fallback demo slots.")
        return ["Tomorrow 10:00 AM", "Tomorrow 4:00 PM", "Wednesday 11:30 AM (Demo)"]

    try:
        SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
        creds = Credentials.from_service_account_file('service_account.json', scopes=SCOPES)
        service = build('calendar', 'v3', credentials=creds)
        calendar_id = os.getenv("LAWYER_CALENDAR_ID", "primary")

        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=4)).isoformat()

        events_result = service.events().list(
            calendarId=calendar_id, timeMin=time_min, timeMax=time_max,
            singleEvents=True, orderBy='startTime').execute()
        events = events_result.get('items', [])

        available_slots = []
        for day_offset in range(1, 4):
            target_date = now + timedelta(days=day_offset)
            if target_date.weekday() >= 5: # Skip weekends (Saturday=5, Sunday=6)
                continue
                
            for hour in [10, 14, 16]: # Default slots: 10 AM, 2 PM, 4 PM
                slot_start = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                slot_end = slot_start + timedelta(hours=1)
                
                conflict = False
                for event in events:
                    start_str = event['start'].get('dateTime', event['start'].get('date'))
                    end_str = event['end'].get('dateTime', event['end'].get('date'))
                    
                    if 'T' not in start_str: 
                        continue # Skip all-day events
                        
                    ev_start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    ev_end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    
                    if not (slot_end <= ev_start or slot_start >= ev_end):
                        conflict = True
                        break
                
                if not conflict:
                    available_slots.append(slot_start.strftime("%A %I:%M %p"))
        
        return available_slots if available_slots else ["No free slots available in the next few days."]
    
    except Exception as e:
        logger.error(f"Calendar Error: {e}")
        return ["Error connecting to calendar system."]

# ==========================================
# AGENT 1: THE ROUTER
# ==========================================
def get_intent(user_message: str, history: List[HistoryItem]) -> str:
    """Micro-prompt to route the query cheaply."""
    history_text = format_history(history)
    prompt = f"""
    CHAT HISTORY:
    {history_text}
    
    NEW MESSAGE: '{user_message}'
    
    TASK: Based on the conversation history and the new message, is the user asking about a CASE or trying to SCHEDULE/confirm a meeting? 
    Reply with exactly one word: CASE or SCHEDULE.
    """
    try:
        response = model.generate_content(prompt)
        intent = response.text.strip().upper()
        return "SCHEDULE" if "SCHEDULE" in intent else "CASE"
    except Exception as e:
        logger.error(f"Router Error: {e}")
        return "CASE"

# ==========================================
# AGENT 2: THE SCHEDULER
# ==========================================
def handle_scheduling(user_message: str, client_id: str, history: List[HistoryItem]):
    """Handles appointment booking with real Google Calendar data."""
    real_slots = get_available_slots()
    slots_str = ", ".join(real_slots)
    history_text = format_history(history)
    
    schema = {
        "type": "OBJECT",
        "properties": {
            "reply_malayalam": {"type": "STRING"},
            "is_confirmed": {"type": "BOOLEAN"},
            "booked_slot": {"type": "STRING"}
        }
    }
    
    prompt = f"""
    CHAT HISTORY: {history_text}
    NEW MESSAGE: {user_message}
    AVAILABLE SLOTS: {slots_str}
    
    TASK: Help user book a slot in Malayalam. Understand context (e.g., if they say 'the first one', look at the history to see what was offered). 
    If they selected a valid slot, set is_confirmed to true and extract the booked_slot exactly as it is written in AVAILABLE SLOTS.
    """
    
    response = model.generate_content(
        prompt, 
        generation_config={"response_mime_type": "application/json", "response_schema": schema}
    )
    data = json.loads(response.text)
    
    # Save to MongoDB if confirmed
    if data.get("is_confirmed") and data.get("booked_slot") and appointments_collection is not None:
        appointments_collection.insert_one({
            "client_id": client_id, 
            "slot": data.get("booked_slot"), 
            "status": "booked",
            "created_at": datetime.now()
        })
        logger.info(f"Booked {data.get('booked_slot')} for {client_id}")
        
    return {"response": data.get("reply_malayalam")}

# ==========================================
# AGENT 3: THE CASE LAWYER
# ==========================================
def handle_case_query(user_message: str, client_id: str, history: List[HistoryItem]):
    """Handles case details retrieval safely."""
    case_context = "No data."
    if clients_collection is not None:
        client_data = clients_collection.find_one({"client_id": client_id}, {"_id": 0})
        if client_data:
            case_context = str(client_data)
            
    history_text = format_history(history)
    
    schema = {
        "type": "OBJECT",
        "properties": {
            "reply_malayalam": {"type": "STRING"},
            "found_in_context": {"type": "BOOLEAN"}
        }
    }
    
    prompt = f"""
    CHAT HISTORY: {history_text}
    CASE FILE: {case_context}
    NEW MESSAGE: {user_message}
    
    TASK: Answer user in Malayalam ONLY using the Case File. Use the chat history to understand the context of their question.
    If the info is missing from the Case File or requires general legal advice, set found_in_context to false.
    """
    
    response = model.generate_content(
        prompt, 
        generation_config={"response_mime_type": "application/json", "response_schema": schema}
    )
    data = json.loads(response.text)
    
    if not data.get("found_in_context"):
        return {"response": "ക്ഷമിക്കണം, നിങ്ങളുടെ കേസ് ഫയലിലുള്ള കാര്യങ്ങൾ മാത്രമേ എനിക്ക് പറയാൻ കഴിയൂ, നിയമപരമായ ഉപദേശങ്ങൾ നൽകാൻ എനിക്ക് അധികാരമില്ല."}
    
    return {"response": data.get("reply_malayalam")}

# ==========================================
# MAIN API ENDPOINT
# ==========================================
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Fast Routing using Memory
        intent = get_intent(request.message, request.history)
        logger.info(f"Detected Intent: {intent} for Client: {request.client_id}")
        
        # 2. Directed Execution
        if intent == "SCHEDULE":
            return handle_scheduling(request.message, request.client_id, request.history)
        else:
            return handle_case_query(request.message, request.client_id, request.history)
            
    except Exception as e:
        logger.error(f"Endpoint Error: {repr(e)}")
        return {"response": "ക്ഷമിക്കണം, ഒരു സാങ്കേതിക തകരാർ ഉണ്ട്. ദയവായി അല്പം കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)