# Sereni - AI Sentiment Analysis Chatbot for Mental Health Support

A compassionate AI-powered chatbot designed for emotional support and suicide prevention awareness. Built as an academic project to demonstrate the potential of AI in mental health support.

> **Disclaimer**: This is an academic project for mental health awareness. It is NOT a substitute for professional medical advice, diagnosis, or treatment. If you're in crisis, please contact emergency services or a mental health helpline immediately.

## Features

- **AI-Powered Chat**: Emotionally intelligent conversations using GPT-4o-mini
- **Sentiment Analysis**: Real-time detection of emotional state and risk levels
- **Crisis Protocol**: Automatic detection of high-risk messages with helpline information
- **5-4-3-2-1 Grounding Exercise**: Interactive sensory grounding technique for anxiety relief
- **Chat History**: Persistent conversation storage with easy management
- **User Authentication**: Secure JWT-based authentication system
- **Responsive Design**: Beautiful, calming UI that works on desktop and mobile

## Tech Stack

- **Frontend**: React, Tailwind CSS, Shadcn UI
- **Backend**: FastAPI (Python)
- **Database**: MongoDB
- **AI**: OpenAI GPT-4o-mini

## Project Structure

```
sereni-chatbot/
├── frontend/                # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── context/        # Auth context
│   │   ├── pages/          # Page components
│   │   └── App.js          # Main app
│   └── package.json
├── backend/                 # FastAPI backend
│   ├── server.py           # Main API server
│   ├── requirements.txt    # Python dependencies
│   └── .env.example        # Environment template
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Setup Instructions

### Prerequisites

- Node.js (v18+)
- Python (v3.9+)
- MongoDB
- OpenAI API Key

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env if needed

# Run the server
uvicorn server:app --reload --host 0.0.0.0 --port 8001
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
yarn install

# Create .env file
cp .env.example .env

# Run the development server
yarn start
```

Access at: `http://localhost:3000/sereni-chatbot`

## Environment Variables

### Backend (.env)

```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=sereni_db
CORS_ORIGINS=*
OPENAI_API_KEY=your_openai_api_key_here
JWT_SECRET=your_secret_key_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

### Frontend (.env)

```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create new account |
| POST | `/api/auth/login` | User login |
| GET | `/api/auth/me` | Get current user |
| POST | `/api/chat` | Send message, get AI response |
| GET | `/api/conversations` | List user conversations |
| GET | `/api/conversations/{id}/messages` | Get conversation messages |
| DELETE | `/api/conversations/{id}` | Delete conversation |
| POST | `/api/grounding/log` | Log grounding exercise |

## Risk Detection

The system analyzes messages for emotional content:

- **Normal**: General conversation
- **Distress**: Mild emotional distress detected
- **Moderate**: Elevated concern, gentle support provided
- **High**: Crisis-level content triggers immediate support resources

## Crisis Helplines (India)

- **iCall**: 9152987821
- **Vandrevala Foundation**: 1860-2662-345
- **AASRA**: 9820466726

## Docker Deployment

```bash
docker-compose up --build
```

Access at: `http://localhost/sereni-chatbot`

## License

MIT License

---

**Remember**: If you or someone you know is struggling, please reach out to a mental health professional or crisis helpline. You are not alone.
