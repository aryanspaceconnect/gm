import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types';
import { SendIcon } from './Icons';
import { parseMarkdown } from '../utils/markdown';

interface ChatInterfaceProps {
  history: ChatMessage[];
  isReplying: boolean;
  onSendMessage: (message: string) => void;
  inputRef: React.RefObject<HTMLInputElement>;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ history, isReplying, onSendMessage, inputRef }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [history, isReplying]);

  // Sync external changes to the input field
  useEffect(() => {
    if (inputRef.current) {
        setInput(inputRef.current.value);
    }
  }, [inputRef.current?.value]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSendMessage(input);
    setInput('');
  };
  
  return (
    <div className="flex flex-col h-full p-4">
      <div className="flex-grow overflow-y-auto pr-2 space-y-4">
        {history.map((msg, index) => (
          <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-xl p-3 rounded-lg ${
                msg.role === 'user' ? 'bg-cyan-800 text-white' : 'bg-gray-700 text-gray-300'
              }`}
            >
             <div className="prose prose-invert prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.text) }} />
            </div>
          </div>
        ))}
        {isReplying && (
             <div className="flex justify-start">
                <div className="max-w-xl p-3 rounded-lg bg-gray-700 text-gray-300">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                    </div>
                </div>
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="flex-shrink-0 pt-4 mt-4 border-t border-gray-700">
        <div className="flex items-center bg-gray-700 rounded-lg pr-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a follow-up question..."
            disabled={isReplying}
            className="w-full bg-transparent p-3 text-gray-200 placeholder-gray-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={isReplying || !input.trim()}
            className="p-2 text-cyan-400 rounded-full hover:bg-gray-600 disabled:text-gray-600 disabled:cursor-not-allowed transition-colors"
          >
            <SendIcon className="h-5 w-5" />
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;
