import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SendIcon } from "lucide-react";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface ChatPanelProps {
  repoId?: string;
}

export default function ChatPanel({ repoId }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage: ChatMessage = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Mock API response - simulate 1-2 second delay
    setTimeout(() => {
      // Add mock assistant response
      const mockResponses = [
        "Based on the repository structure, this appears to be using a combination of Next.js for the frontend and FastAPI for the backend.",
        "This project follows a modular architecture with clear separation between frontend and API components.",
        "The repository contains RESTful API endpoints that handle data processing and retrieval operations.",
        "From what I can see, this project implements a Retrieval Augmented Generation (RAG) pattern for handling documentation.",
        "The codebase appears to use modern React patterns including hooks and functional components."
      ];
      
      const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];
      const assistantMessage: ChatMessage = { 
        role: "assistant", 
        content: `${randomResponse} Let me know if you have any other questions about the architecture or implementation details.` 
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 1000 + Math.random() * 1000);
  };

  return (
    <Card className="border-t border-gray-200 p-3 flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <div className="text-base font-semibold">Ask about this repository</div>
        <div className="text-xs text-gray-500">AI-powered assistant</div>
      </div>
      
      <div className="flex-1 overflow-auto mb-2 max-h-[200px] space-y-2">
        {messages.length === 0 ? (
          <div className="text-sm text-gray-400 italic">
            No messages yet. Start the conversation by asking a question.
          </div>
        ) : (
          messages.map((message, index) => (
            <div 
              key={index} 
              className={`p-2 rounded-lg ${
                message.role === "user" 
                  ? "bg-blue-50 ml-6 border border-blue-100" 
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
          className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          disabled={isLoading}
        />
        <Button 
          type="submit" 
          disabled={isLoading || !input.trim()}
          size="sm"
          className="bg-blue-600 hover:bg-blue-700 h-7 w-7 p-0"
        >
          <SendIcon className="w-3 h-3" />
        </Button>
      </form>
    </Card>
  );
} 