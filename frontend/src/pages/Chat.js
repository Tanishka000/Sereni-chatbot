import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { ScrollArea } from '../components/ui/scroll-area';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '../components/ui/dialog';
import { 
  MessageCircleHeart, 
  SendHorizontal, 
  Menu, 
  Plus, 
  History, 
  Wind, 
  LifeBuoy, 
  LogOut,
  Trash2,
  Loader2,
  X
} from 'lucide-react';
import { toast } from 'sonner';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Grounding Exercise Steps
const GROUNDING_STEPS = [
  { count: 5, sense: 'SEE', instruction: 'Look around and name 5 things you can see.', emoji: '👁️' },
  { count: 4, sense: 'TOUCH', instruction: 'Notice 4 things you can physically feel.', emoji: '✋' },
  { count: 3, sense: 'HEAR', instruction: 'Listen for 3 sounds around you.', emoji: '👂' },
  { count: 2, sense: 'SMELL', instruction: 'Identify 2 things you can smell.', emoji: '👃' },
  { count: 1, sense: 'TASTE', instruction: 'Notice 1 thing you can taste.', emoji: '👅' },
];

const TypingIndicator = () => (
  <div className="flex items-center gap-1.5 px-4 py-3">
    <div className="w-2 h-2 rounded-full bg-primary/60 typing-dot" />
    <div className="w-2 h-2 rounded-full bg-primary/60 typing-dot" />
    <div className="w-2 h-2 rounded-full bg-primary/60 typing-dot" />
  </div>
);

const ChatBubble = ({ message, isUser }) => {
  const showCrisisAlert = message.risk_level === 'high' && isUser;
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}>
      <div
        className={`max-w-[85%] md:max-w-[75%] px-4 py-3 ${
          isUser
            ? 'bg-primary text-primary-foreground chat-bubble-user'
            : 'bg-white border border-border/50 text-foreground chat-bubble-ai shadow-soft'
        }`}
      >
        <p className="text-sm md:text-base leading-relaxed whitespace-pre-wrap">{message.content}</p>
        {showCrisisAlert && (
          <div className="mt-2 pt-2 border-t border-primary-foreground/20 text-xs opacity-80">
            Support is available
          </div>
        )}
      </div>
    </div>
  );
};

const ConversationItem = ({ conversation, isActive, onClick, onDelete }) => {
  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <div
      className={`group flex items-center gap-2 px-3 py-2.5 rounded-xl cursor-pointer transition-all ${
        isActive ? 'bg-primary/10 text-primary' : 'hover:bg-muted'
      }`}
      onClick={onClick}
      data-testid={`conversation-${conversation.id}`}
    >
      <History className="w-4 h-4 flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{conversation.title}</p>
        <p className="text-xs text-muted-foreground">{formatDate(conversation.updated_at)}</p>
      </div>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onDelete(conversation.id);
        }}
        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/10 rounded-lg transition-all"
        data-testid={`delete-conversation-${conversation.id}`}
      >
        <Trash2 className="w-3.5 h-3.5 text-destructive" />
      </button>
    </div>
  );
};

const GroundingModal = ({ open, onClose }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  const handleNext = () => {
    if (currentStep < GROUNDING_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      setIsComplete(true);
      // Log grounding completion
      axios.post(`${API}/grounding/log`, { completed: true }).catch(console.error);
    }
  };

  const handleClose = () => {
    setCurrentStep(0);
    setIsComplete(false);
    onClose();
  };

  const step = GROUNDING_STEPS[currentStep];

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-lg glass-card border-border/40" data-testid="grounding-modal">
        <DialogHeader>
          <DialogTitle className="font-heading text-2xl text-center flex items-center justify-center gap-2">
            <Wind className="w-6 h-6 text-primary" />
            Pause & Breathe
          </DialogTitle>
          <DialogDescription className="text-center">
            Let's ground yourself with the 5-4-3-2-1 technique
          </DialogDescription>
        </DialogHeader>
        
        <div className="py-8">
          {!isComplete ? (
            <div className="text-center space-y-6 animate-fade-in" key={currentStep}>
              <div className="space-y-2">
                <div className="text-5xl mb-4">{step.emoji}</div>
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10">
                  <span className="text-3xl font-bold text-primary">{step.count}</span>
                  <span className="text-lg font-semibold text-primary">things you can {step.sense}</span>
                </div>
                <p className="text-muted-foreground mt-4">{step.instruction}</p>
              </div>
              
              <div className="flex items-center justify-center gap-2 pt-4">
                {GROUNDING_STEPS.map((_, idx) => (
                  <div
                    key={idx}
                    className={`w-2 h-2 rounded-full transition-colors ${
                      idx === currentStep ? 'bg-primary' : idx < currentStep ? 'bg-primary/40' : 'bg-muted'
                    }`}
                  />
                ))}
              </div>
              
              <Button
                onClick={handleNext}
                className="rounded-full px-8 py-6 text-base font-semibold"
                data-testid="grounding-next-btn"
              >
                {currentStep < GROUNDING_STEPS.length - 1 ? 'Next' : 'Complete'}
              </Button>
            </div>
          ) : (
            <div className="text-center space-y-6 animate-fade-in">
              <div className="w-20 h-20 mx-auto rounded-full bg-primary/10 flex items-center justify-center">
                <span className="text-4xl">💚</span>
              </div>
              <div>
                <h3 className="font-heading text-xl font-bold text-foreground mb-2">
                  You did something kind for yourself.
                </h3>
                <p className="text-muted-foreground">
                  Take a moment to appreciate this small act of self-care.
                </p>
              </div>
              <Button
                onClick={handleClose}
                className="rounded-full px-8"
                data-testid="grounding-close-btn"
              >
                Continue
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

const CrisisPanel = ({ show, onClose }) => {
  if (!show) return null;
  
  return (
    <div className="bg-crisis-background border-b-2 border-crisis/30 px-4 py-3 animate-fade-in" data-testid="crisis-panel">
      <div className="max-w-4xl mx-auto flex items-start gap-4">
        <LifeBuoy className="w-5 h-5 text-crisis flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="text-sm font-semibold text-foreground mb-1">Support is available for you</p>
          <p className="text-xs text-muted-foreground mb-2">
            If you're going through a difficult time, please consider reaching out:
          </p>
          <div className="flex flex-wrap gap-3 text-xs">
            <span className="px-2 py-1 rounded-full bg-white border border-crisis/20">
              <strong>iCall:</strong> 9152987821
            </span>
            <span className="px-2 py-1 rounded-full bg-white border border-crisis/20">
              <strong>Vandrevala Foundation:</strong> 1860-2662-345
            </span>
            <span className="px-2 py-1 rounded-full bg-white border border-crisis/20">
              <strong>AASRA:</strong> 9820466726
            </span>
          </div>
        </div>
        <button onClick={onClose} className="p-1 hover:bg-white/50 rounded-lg transition-colors">
          <X className="w-4 h-4 text-muted-foreground" />
        </button>
      </div>
    </div>
  );
};

// Mobile Sidebar Component - Custom implementation without overlay blocking
const MobileSidebar = ({ isOpen, onClose, children }) => {
  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden transition-opacity"
          onClick={onClose}
          data-testid="mobile-sidebar-backdrop"
        />
      )}
      {/* Sidebar */}
      <div 
        className={`fixed inset-y-0 left-0 w-72 z-50 bg-background glass border-r border-white/20 transform transition-transform duration-300 ease-in-out md:hidden ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
        data-testid="mobile-sidebar"
      >
        <button 
          onClick={onClose}
          className="absolute right-3 top-3 p-2 rounded-lg hover:bg-muted transition-colors"
          data-testid="mobile-sidebar-close"
        >
          <X className="w-4 h-4" />
        </button>
        {children}
      </div>
    </>
  );
};

const Chat = () => {
  const navigate = useNavigate();
  const { user, logout, isAuthenticated, loading: authLoading } = useAuth();
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isFetchingHistory, setIsFetchingHistory] = useState(false);
  const [showGrounding, setShowGrounding] = useState(false);
  const [showCrisisPanel, setShowCrisisPanel] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      navigate('/');
    }
  }, [authLoading, isAuthenticated, navigate]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchConversations();
    }
  }, [isAuthenticated]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchConversations = async () => {
    try {
      const response = await axios.get(`${API}/conversations`);
      setConversations(response.data);
    } catch (error) {
      console.error('Failed to fetch conversations:', error);
    }
  };

  const loadConversation = async (conversationId) => {
    setIsFetchingHistory(true);
    setSidebarOpen(false);
    try {
      const response = await axios.get(`${API}/conversations/${conversationId}/messages`);
      setMessages(response.data);
      setCurrentConversationId(conversationId);
      
      // Check if any message has high risk
      const hasHighRisk = response.data.some(m => m.risk_level === 'high');
      setShowCrisisPanel(hasHighRisk);
    } catch (error) {
      toast.error('Failed to load conversation');
    } finally {
      setIsFetchingHistory(false);
    }
  };

  const startNewConversation = () => {
    setCurrentConversationId(null);
    setMessages([]);
    setShowCrisisPanel(false);
    setSidebarOpen(false);
    inputRef.current?.focus();
  };

  const deleteConversation = async (id) => {
    try {
      await axios.delete(`${API}/conversations/${id}`);
      setConversations(prev => prev.filter(c => c.id !== id));
      if (currentConversationId === id) {
        startNewConversation();
      }
      toast.success('Conversation deleted');
    } catch (error) {
      toast.error('Failed to delete conversation');
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const content = inputValue.trim();
    setInputValue('');
    setIsLoading(true);

    // Optimistically add user message
    const tempUserMsg = {
      id: 'temp-user',
      content,
      role: 'user',
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, tempUserMsg]);

    try {
      const response = await axios.post(`${API}/chat`, {
        content,
        conversation_id: currentConversationId
      });

      const { conversation_id, user_message, ai_message } = response.data;
      
      // Update messages (replace temp with real)
      setMessages(prev => [
        ...prev.filter(m => m.id !== 'temp-user'),
        user_message,
        ai_message
      ]);
      
      setCurrentConversationId(conversation_id);
      
      // Show crisis panel if high risk detected
      if (user_message.risk_level === 'high') {
        setShowCrisisPanel(true);
      }
      
      // Refresh conversations list
      fetchConversations();
    } catch (error) {
      toast.error('Failed to send message. Please try again.');
      setMessages(prev => prev.filter(m => m.id !== 'temp-user'));
      setInputValue(content);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    setSidebarOpen(false);
    logout();
    navigate('/');
  };

  const handleGroundingClick = () => {
    setSidebarOpen(false);
    setShowGrounding(true);
  };

  const handleSupportClick = () => {
    setSidebarOpen(false);
    setShowCrisisPanel(true);
  };

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  const SidebarContent = () => (
    <div className="h-full flex flex-col pt-2">
      {/* Header */}
      <div className="p-4 border-b border-border/40">
        <div className="flex items-center gap-2 mb-4">
          <MessageCircleHeart className="w-6 h-6 text-primary" strokeWidth={1.5} />
          <span className="font-heading font-bold text-lg">Sereni</span>
        </div>
        <Button
          onClick={startNewConversation}
          className="w-full rounded-full justify-center gap-2"
          data-testid="new-chat-btn"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </Button>
      </div>
      
      {/* Conversations */}
      <ScrollArea className="flex-1 p-2">
        <div className="space-y-1">
          {conversations.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">
              No conversations yet
            </p>
          ) : (
            conversations.map(conv => (
              <ConversationItem
                key={conv.id}
                conversation={conv}
                isActive={conv.id === currentConversationId}
                onClick={() => loadConversation(conv.id)}
                onDelete={deleteConversation}
              />
            ))
          )}
        </div>
      </ScrollArea>
      
      {/* Footer */}
      <div className="p-4 border-t border-border/40 space-y-2">
        <Button
          variant="ghost"
          className="w-full justify-start gap-2 rounded-xl text-muted-foreground hover:text-foreground"
          onClick={handleGroundingClick}
          data-testid="grounding-btn"
        >
          <Wind className="w-4 h-4" />
          Pause & Breathe
        </Button>
        <Button
          variant="ghost"
          className="w-full justify-start gap-2 rounded-xl text-muted-foreground hover:text-foreground"
          onClick={handleSupportClick}
          data-testid="get-support-btn"
        >
          <LifeBuoy className="w-4 h-4" />
          Get Support
        </Button>
        <div className="pt-2 border-t border-border/40">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium truncate">{user?.name}</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={handleLogout}
              data-testid="logout-btn"
            >
              <LogOut className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen flex bg-background" data-testid="chat-page">
      {/* Desktop Sidebar */}
      <aside className="hidden md:flex w-72 flex-col glass border-r border-white/20">
        <SidebarContent />
      </aside>

      {/* Mobile Sidebar - Custom implementation */}
      <MobileSidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)}>
        <SidebarContent />
      </MobileSidebar>

      {/* Main Content */}
      <main className="flex-1 flex flex-col">
        {/* Crisis Panel */}
        <CrisisPanel show={showCrisisPanel} onClose={() => setShowCrisisPanel(false)} />
        
        {/* Header (Mobile) */}
        <header className="md:hidden flex items-center gap-3 px-4 py-3 border-b border-border/40 bg-white/50">
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9"
            onClick={() => setSidebarOpen(true)}
            data-testid="mobile-menu-btn"
          >
            <Menu className="w-5 h-5" />
          </Button>
          <div className="flex items-center gap-2">
            <MessageCircleHeart className="w-5 h-5 text-primary" strokeWidth={1.5} />
            <span className="font-heading font-bold">Sereni</span>
          </div>
        </header>

        {/* Messages Area */}
        <ScrollArea className="flex-1 px-4 py-6">
          <div className="max-w-3xl mx-auto space-y-4">
            {messages.length === 0 && !isFetchingHistory && (
              <div className="text-center py-16 animate-fade-in">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                  <MessageCircleHeart className="w-8 h-8 text-primary" strokeWidth={1.5} />
                </div>
                <h2 className="font-heading text-xl font-bold text-foreground mb-2">
                  Hi{user?.name ? `, ${user.name.split(' ')[0]}` : ''}! How are you feeling today?
                </h2>
                <p className="text-muted-foreground max-w-md mx-auto">
                  I'm Sereni, your compassionate companion. Share what's on your mind – I'm here to listen without judgment.
                </p>
              </div>
            )}
            
            {isFetchingHistory && (
              <div className="flex justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-primary" />
              </div>
            )}
            
            {messages.map((message) => (
              <ChatBubble key={message.id} message={message} isUser={message.role === 'user'} />
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-border/50 chat-bubble-ai shadow-soft">
                  <TypingIndicator />
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div className="border-t border-border/40 bg-white/50 px-4 py-4">
          <form onSubmit={sendMessage} className="max-w-3xl mx-auto">
            <div className="flex items-center gap-2">
              <Input
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Type your message..."
                className="flex-1 rounded-full px-5 py-6 bg-white border-border/50 focus:ring-primary/20"
                disabled={isLoading}
                data-testid="chat-input"
              />
              <Button
                type="submit"
                size="icon"
                className="h-12 w-12 rounded-full"
                disabled={!inputValue.trim() || isLoading}
                data-testid="send-message-btn"
              >
                <SendHorizontal className="w-5 h-5" />
              </Button>
            </div>
          </form>
          
          {/* Disclaimer */}
          <p className="text-center text-xs text-muted-foreground mt-3 max-w-lg mx-auto">
            Academic project for mental health awareness. Not a substitute for professional help.
          </p>
        </div>
      </main>

      {/* Grounding Modal */}
      <GroundingModal open={showGrounding} onClose={() => setShowGrounding(false)} />
    </div>
  );
};

export default Chat;
