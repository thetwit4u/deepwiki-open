import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SendIcon } from "lucide-react";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface ChatPanelProps {
  repoId?: string;
  collectionName?: string;
}

export default function ChatPanel({ repoId, collectionName }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Load chat history when repoId changes
  useEffect(() => {
    if (!repoId) return;
    
    const fetchChatHistory = async () => {
      setIsLoadingHistory(true);
      try {
        const response = await fetch(`/api/chat?repoId=${encodeURIComponent(repoId)}`);
        if (response.ok) {
          const data = await response.json();
          if (data.messages && Array.isArray(data.messages)) {
            setMessages(data.messages);
          }
        } else {
          console.error("Failed to load chat history:", await response.text());
        }
      } catch (error) {
        console.error("Error loading chat history:", error);
      } finally {
        setIsLoadingHistory(false);
      }
    };
    
    fetchChatHistory();
  }, [repoId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !repoId) return;

    // Add user message immediately
    const userMessage: ChatMessage = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // Prepare request payload with optional collection name
      const payload: any = {
        repoId,
        message: input,
      };
      
      // Add collection name if provided
      if (collectionName) {
        console.log(`Using specific collection name: ${collectionName}`);
        payload.collectionName = collectionName;
      }
      
      // Call the API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`API responded with status: ${response.status}`);
      }

      const data = await response.json();
      
      // Add assistant response
      const assistantMessage: ChatMessage = { 
        role: "assistant", 
        content: data.answer || "I couldn't generate a response based on the repository content."
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error calling chat API:", error);
      
      // Add error message
      const errorMessage: ChatMessage = { 
        role: "assistant", 
        content: "I encountered an error while processing your request. Please try again." 
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="border-t border-gray-200 p-3 flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <div className="text-base font-semibold">Ask about this repository</div>
        <div className="text-xs text-gray-500">AI-powered assistant</div>
      </div>
      
      <div className="flex-1 overflow-auto mb-2 max-h-[200px] space-y-2">
        {isLoadingHistory ? (
          <div className="text-sm text-gray-400 italic">
            Loading chat history...
          </div>
        ) : messages.length === 0 ? (
          <div className="text-sm text-gray-400 italic">
            No messages yet. Start the conversation by asking a question.
          </div>
        ) : (
          messages.map((message, index) => (
            <div 
              key={index} 
              className={`p-2 rounded-lg ${
                message.role === "user" 
                  ? "bg-gray-100 ml-6 border border-gray-200" 
                  : "bg-gray-50 mr-6 border border-gray-100"
              }`}
            >
              <div className="text-xs font-semibold mb-1">
                {message.role === "user" ? "You" : "AI Assistant"}
              </div>
              <div className="text-xs">{message.content}</div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="p-2 rounded-lg bg-gray-50 mr-6 border border-gray-100">
            <div className="text-xs font-semibold mb-1">AI Assistant</div>
            <div className="flex items-center space-x-2">
              <div className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-pulse"></div>
              <div className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-pulse delay-100"></div>
              <div className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-pulse delay-200"></div>
            </div>
          </div>
        )}
      </div>
      
      <form onSubmit={handleSubmit} className="flex space-x-1">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about this codebase..."
          className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-gray-500"
          disabled={isLoading || !repoId}
        />
        <Button 
          type="submit" 
          disabled={isLoading || !input.trim() || !repoId}
          size="sm"
          className="bg-gray-800 hover:bg-black h-7 w-7 p-0"
        >
          <SendIcon className="w-3 h-3" />
        </Button>
      </form>
    </Card>
  );
} 