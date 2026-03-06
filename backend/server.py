from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt
from openai import OpenAI

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# OpenAI client
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# JWT Config
JWT_SECRET = os.environ.get('JWT_SECRET', 'sereni_secret')
JWT_ALGORITHM = os.environ.get('JWT_ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES', 1440))

# Create the main app
app = FastAPI(title="Sereni API", description="AI Sentiment Analysis Chatbot for Mental Health Support")

# Create router with /api prefix
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class MessageCreate(BaseModel):
    content: str
    conversation_id: Optional[str] = None

class MessageResponse(BaseModel):
    id: str
    content: str
    role: str
    sentiment: Optional[str] = None
    risk_level: Optional[str] = None
    timestamp: str

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int

class ChatResponse(BaseModel):
    conversation_id: str
    user_message: MessageResponse
    ai_message: MessageResponse

class GroundingLogCreate(BaseModel):
    completed: bool = True

# ==================== RISK DETECTION ====================

HIGH_RISK_PHRASES = [
    "i want to kill myself",
    "i don't want to live",
    "i want to die",
    "kill myself",
    "end my life",
    "suicide",
    "better off dead",
    "no reason to live",
    "can't go on",
    "want to end it"
]

DISTRESS_KEYWORDS = [
    "hopeless", "worthless", "alone", "nobody cares", "can't cope",
    "exhausted", "overwhelmed", "broken", "empty", "numb",
    "trapped", "burden", "failing", "hate myself", "useless"
]

MODERATE_KEYWORDS = [
    "sad", "anxious", "worried", "stressed", "scared",
    "lonely", "frustrated", "angry", "upset", "confused",
    "tired", "struggling", "difficult", "hard time"
]

def analyze_sentiment_and_risk(message: str) -> tuple:
    """Analyze message for sentiment and risk level."""
    message_lower = message.lower()
    
    for phrase in HIGH_RISK_PHRASES:
        if phrase in message_lower:
            return "crisis", "high"
    
    distress_count = sum(1 for kw in DISTRESS_KEYWORDS if kw in message_lower)
    moderate_count = sum(1 for kw in MODERATE_KEYWORDS if kw in message_lower)
    
    if distress_count >= 3:
        return "negative", "high"
    elif distress_count >= 1:
        return "negative", "moderate"
    elif moderate_count >= 2:
        return "mixed", "distress"
    elif moderate_count >= 1:
        return "mixed", "normal"
    else:
        return "neutral", "normal"

def get_system_prompt(risk_level: str) -> str:
    """Get dynamic system prompt based on risk level."""
    base_prompt = """You are Sereni, a super friendly and caring AI buddy who's always here to chat!

YOUR PERSONALITY:
- You're like a warm, supportive best friend who genuinely cares
- You use a casual, friendly tone - like texting a close friend
- You're cheerful but never fake - you match the user's energy
- You use gentle humor when appropriate to lighten the mood
- You celebrate small wins and encourage self-compassion
- You're curious and ask thoughtful questions about their day/feelings
- You use encouraging phrases like "I hear you", "That makes total sense", "I'm so glad you shared that"

CONVERSATION STYLE:
- Keep responses warm and conversational, not robotic or clinical
- Use proper grammar but keep it natural and flowing
- Occasionally use friendly expressions like "Hey!", "Oh wow", "I totally get that"
- Be genuine - if something sounds tough, acknowledge it with empathy
- End messages with something supportive or a gentle question
- Make the person feel heard, valued, and less alone

IMPORTANT REMINDERS:
- You're NOT a replacement for professional mental health care
- This is an ACADEMIC PROJECT for mental health awareness
- Never diagnose or prescribe anything
- Gently encourage professional help when it seems needed"""

    if risk_level == "high":
        return base_prompt + """

GENTLE CRISIS SUPPORT MODE:
- The user might be going through something really hard right now
- Be extra gentle, warm, and present with them
- Softly ask if they're feeling safe - no pressure, just care
- Let them know they're not alone and you're right here with them
- Gently mention that talking to someone they trust could really help
- Share these helplines naturally: iCall (9152987821), Vandrevala Foundation (1860-2662-345)
- Stay calm and reassuring - don't panic or alarm them
- Focus on being a comforting presence"""

    elif risk_level == "moderate":
        return base_prompt + """

EXTRA SUPPORTIVE MODE:
- Sounds like they're having a rough time - be extra caring
- Really listen and validate what they're going through
- Let them know it's okay to feel this way
- Gently explore what might help them feel a bit better
- Remind them that reaching out to someone they trust is always okay
- Be their cheerleader while being realistic"""

    elif risk_level == "distress":
        return base_prompt + """

CARING FRIEND MODE:
- They seem a bit stressed or down - be a supportive buddy
- Help them feel heard and understood
- Maybe suggest some gentle self-care ideas if it feels right
- Remind them to be kind to themselves
- You're here to help them process and feel better"""

    else:
        return base_prompt + """

FRIENDLY CHAT MODE:
- Have a fun, warm conversation!
- Be curious about their life, day, thoughts
- Share in their joy when good things happen
- Be playful and light when the vibe is right
- Help them reflect on the good stuff
- Just be a great friend to chat with!"""

# ==================== AUTH HELPERS ====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await db.users.find_one({"id": user_id}, {"_id": 0})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "password_hash": hash_password(user_data.password),
        "created_at": now
    }
    
    await db.users.insert_one(user_doc)
    token = create_access_token(user_id, user_data.email)
    
    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user_id, email=user_data.email, name=user_data.name, created_at=now)
    )

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_access_token(user["id"], user["email"])
    
    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user["id"], email=user["email"], name=user["name"], created_at=user["created_at"])
    )

@api_router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user["name"],
        created_at=current_user["created_at"]
    )

# ==================== CHAT ROUTES ====================

@api_router.post("/chat", response_model=ChatResponse)
async def send_message(message_data: MessageCreate, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    now = datetime.now(timezone.utc)
    
    if message_data.conversation_id:
        conversation = await db.conversations.find_one(
            {"id": message_data.conversation_id, "user_id": user_id},
            {"_id": 0}
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = message_data.conversation_id
    else:
        conversation_id = str(uuid.uuid4())
        title = message_data.content[:50] + "..." if len(message_data.content) > 50 else message_data.content
        conversation = {
            "id": conversation_id,
            "user_id": user_id,
            "title": title,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        await db.conversations.insert_one(conversation)
    
    sentiment, risk_level = analyze_sentiment_and_risk(message_data.content)
    
    user_msg_id = str(uuid.uuid4())
    user_message = {
        "id": user_msg_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "content": message_data.content,
        "role": "user",
        "sentiment": sentiment,
        "risk_level": risk_level,
        "timestamp": now.isoformat()
    }
    await db.messages.insert_one(user_message)
    
    history = await db.messages.find(
        {"conversation_id": conversation_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(20)
    
    system_prompt = get_system_prompt(risk_level)
    openai_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in history[-10:]:
        openai_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            temperature=0.85,
            max_tokens=500
        )
        ai_content = response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        ai_content = "Hey, I'm having a tiny moment here, but I'm still here for you! Your feelings totally matter. Want to try sharing again? I'm all ears!"
    
    ai_msg_id = str(uuid.uuid4())
    ai_timestamp = datetime.now(timezone.utc)
    ai_message = {
        "id": ai_msg_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "content": ai_content,
        "role": "assistant",
        "sentiment": None,
        "risk_level": None,
        "timestamp": ai_timestamp.isoformat()
    }
    await db.messages.insert_one(ai_message)
    
    await db.conversations.update_one(
        {"id": conversation_id},
        {"$set": {"updated_at": ai_timestamp.isoformat()}}
    )
    
    return ChatResponse(
        conversation_id=conversation_id,
        user_message=MessageResponse(
            id=user_msg_id,
            content=message_data.content,
            role="user",
            sentiment=sentiment,
            risk_level=risk_level,
            timestamp=now.isoformat()
        ),
        ai_message=MessageResponse(
            id=ai_msg_id,
            content=ai_content,
            role="assistant",
            sentiment=None,
            risk_level=None,
            timestamp=ai_timestamp.isoformat()
        )
    )

@api_router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    
    conversations = await db.conversations.find(
        {"user_id": user_id},
        {"_id": 0}
    ).sort("updated_at", -1).to_list(100)
    
    result = []
    for conv in conversations:
        msg_count = await db.messages.count_documents({"conversation_id": conv["id"]})
        result.append(ConversationResponse(
            id=conv["id"],
            title=conv["title"],
            created_at=conv["created_at"],
            updated_at=conv["updated_at"],
            message_count=msg_count
        ))
    
    return result

@api_router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(conversation_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    
    conversation = await db.conversations.find_one(
        {"id": conversation_id, "user_id": user_id},
        {"_id": 0}
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = await db.messages.find(
        {"conversation_id": conversation_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(1000)
    
    return [MessageResponse(
        id=msg["id"],
        content=msg["content"],
        role=msg["role"],
        sentiment=msg.get("sentiment"),
        risk_level=msg.get("risk_level"),
        timestamp=msg["timestamp"]
    ) for msg in messages]

@api_router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    
    result = await db.conversations.delete_one({"id": conversation_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    await db.messages.delete_many({"conversation_id": conversation_id})
    
    return {"message": "Conversation deleted successfully"}

# ==================== GROUNDING ROUTES ====================

@api_router.post("/grounding/log")
async def log_grounding(data: GroundingLogCreate, current_user: dict = Depends(get_current_user)):
    log_entry = {
        "id": str(uuid.uuid4()),
        "user_id": current_user["id"],
        "completed": data.completed,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await db.grounding_logs.insert_one(log_entry)
    return {"message": "Grounding exercise logged", "id": log_entry["id"]}

# ==================== HEALTH CHECK ====================

@api_router.get("/")
async def root():
    return {"message": "Sereni API is running", "status": "healthy"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "sereni-api"}

# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
