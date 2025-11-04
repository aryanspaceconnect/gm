import React, { useState, useCallback } from 'react';
import CodeEditor from './components/CodeEditor';
import ResultDisplay from './components/ResultDisplay';
import Header from './components/Header';
import { analyzeCodeAndOutput, askFollowUp, AnalysisData } from './services/geminiService';
import { INITIAL_CODE, MOCK_OUTPUT } from './constants';
import { ChatMessage } from './types';

const App: React.FC = () => {
  const [pythonCode, setPythonCode] = useState<string>(INITIAL_CODE);
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isReplying, setIsReplying] = useState<boolean>(false);

  const handleRunAndAnalyze = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setAnalysisData(null);
    setChatHistory([]);
    
    try {
      const data = await analyzeCodeAndOutput(pythonCode, MOCK_OUTPUT);
      setAnalysisData(data);
      // Seed the chat history with the initial summary
      if (data.summary) {
        setChatHistory([{ role: 'model', text: data.summary }]);
      }
    } catch (err) {
      if (err instanceof Error) {
        setError(`Failed to get analysis: ${err.message}`);
      } else {
        setError('An unknown error occurred.');
      }
    } finally {
      setIsLoading(false);
    }
  }, [pythonCode]);

  const handleSendFollowUp = async (question: string) => {
    if (!question.trim()) return;

    const newHistory: ChatMessage[] = [...chatHistory, { role: 'user', text: question }];
    setChatHistory(newHistory);
    setIsReplying(true);

    try {
      const reply = await askFollowUp(question);
      setChatHistory([...newHistory, { role: 'model', text: reply }]);
    } catch (err) {
       if (err instanceof Error) {
        setChatHistory([...newHistory, { role: 'model', text: `Sorry, I encountered an error: ${err.message}` }]);
      } else {
        setChatHistory([...newHistory, { role: 'model', text: 'Sorry, an unknown error occurred.' }]);
      }
    } finally {
      setIsReplying(false);
    }
  };


  return (
    <div className="min-h-screen bg-gray-900 text-gray-300 font-sans flex flex-col">
      <Header />
      <main className="flex-grow container mx-auto p-4 md:p-6 lg:p-8 flex flex-col lg:flex-row gap-6">
        <div className="lg:w-1/2 flex flex-col h-[85vh]">
          <CodeEditor
            code={pythonCode}
            setCode={setPythonCode}
            onRun={handleRunAndAnalyze}
            isLoading={isLoading}
          />
        </div>
        <div className="lg:w-1/2 flex flex-col h-[85vh]">
          <ResultDisplay
            analysisData={analysisData}
            isLoading={isLoading}
            error={error}
            chatHistory={chatHistory}
            isReplying={isReplying}
            onSendFollowUp={handleSendFollowUp}
          />
        </div>
      </main>
    </div>
  );
};

export default App;
