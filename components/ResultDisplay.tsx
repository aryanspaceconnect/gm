import React, { useState, useEffect, useRef } from 'react';
import { AnalysisData } from '../services/geminiService';
import Loader from './Loader';
import SimulationCanvas from './SimulationCanvas';
import { FitnessChart, MetricsDisplay } from './Charts';
import ChatInterface from './ChatInterface';
import { ChatMessage } from '../types';
import { AskIcon } from './Icons';

interface ResultDisplayProps {
  analysisData: AnalysisData | null;
  isLoading: boolean;
  error: string | null;
  chatHistory: ChatMessage[];
  isReplying: boolean;
  onSendFollowUp: (question: string) => void;
}

type Tab = 'Simulation' | 'Performance' | 'Analysis';

const ResultDisplay: React.FC<ResultDisplayProps> = (props) => {
  const { analysisData, isLoading, error, chatHistory, isReplying, onSendFollowUp } = props;
  const [activeTab, setActiveTab] = useState<Tab>('Analysis');
  const [selectionPopup, setSelectionPopup] = useState<{
    visible: boolean;
    x: number;
    y: number;
    text: string;
  }>({ visible: false, x: 0, y: 0, text: '' });
  const analysisRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (analysisData) {
      setActiveTab('Simulation');
    }
  }, [analysisData]);

  useEffect(() => {
    const handleGlobalClick = () => {
      if (selectionPopup.visible) {
        setSelectionPopup({ ...selectionPopup, visible: false });
      }
    };
    window.addEventListener('mousedown', handleGlobalClick);
    return () => {
      window.removeEventListener('mousedown', handleGlobalClick);
    };
  }, [selectionPopup]);
  
  const handleSelection = () => {
    const selection = window.getSelection();
    const selectedText = selection?.toString().trim() ?? '';
    if (selectedText.length > 10 && analysisRef.current?.contains(selection?.anchorNode)) {
      const range = selection!.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      const containerRect = analysisRef.current.getBoundingClientRect();
      setSelectionPopup({
        visible: true,
        x: rect.left - containerRect.left + rect.width / 2,
        y: rect.top - containerRect.top - 35, // Position above selection
        text: selectedText,
      });
    } else {
      setSelectionPopup({ ...selectionPopup, visible: false });
    }
  };

  const handleAskAboutSelection = () => {
    const question = `Can you elaborate on this part: "${selectionPopup.text}"?`;
    if (chatInputRef.current) {
      chatInputRef.current.value = question;
      chatInputRef.current.focus();
    }
    setSelectionPopup({ ...selectionPopup, visible: false });
    setActiveTab('Analysis');
  };

  const renderContent = () => {
    if (isLoading) return <Loader />;
    if (error) {
      return (
        <div className="p-4 text-red-400 bg-red-900/50 rounded-md">
          <h3 className="font-bold mb-2">An Error Occurred</h3>
          <p>{error}</p>
        </div>
      );
    }
    if (!analysisData) {
      return (
        <div className="text-center text-gray-500 flex flex-col items-center justify-center h-full">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m12.728 0l.707.707M6.343 17.657l.707.707m12.728 0l-.707.707M12 21v-1" />
          </svg>
          <p className="text-lg">Click "Run & Analyze" to see the results.</p>
        </div>
      );
    }

    return (
      <div className="flex flex-col h-full">
        <div className="flex-shrink-0 border-b border-gray-700">
          <nav className="-mb-px flex space-x-4 px-4" aria-label="Tabs">
            {(['Simulation', 'Performance', 'Analysis'] as Tab[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`${
                  activeTab === tab
                    ? 'border-cyan-500 text-cyan-400'
                    : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-500'
                } whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>
        <div className="flex-grow overflow-y-auto relative" ref={analysisRef} onMouseUp={handleSelection}>
          {selectionPopup.visible && (
             <button 
                onClick={handleAskAboutSelection}
                className="absolute z-10 bg-gray-900 border border-cyan-500 text-cyan-400 rounded-full px-3 py-1 text-xs font-semibold flex items-center gap-1 hover:bg-cyan-900/50 transition-all"
                style={{ left: `${selectionPopup.x}px`, top: `${selectionPopup.y}px`, transform: 'translateX(-50%)' }}
                onMouseDown={(e) => e.stopPropagation()} // Prevent global click from hiding it immediately
             >
               <AskIcon className="h-4 w-4" /> Ask Gemini
             </button>
          )}

          {activeTab === 'Simulation' && <SimulationCanvas trajectoryData={analysisData.trajectoryData} />}
          {activeTab === 'Performance' && (
            <div className="p-4 space-y-6">
                <MetricsDisplay metrics={analysisData.finalMetrics} />
                <FitnessChart fitnessData={analysisData.fitnessChartData} />
            </div>
          )}
          {activeTab === 'Analysis' && (
             <ChatInterface 
                history={chatHistory} 
                isReplying={isReplying} 
                onSendMessage={onSendFollowUp}
                inputRef={chatInputRef}
             />
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-800 rounded-lg shadow-lg flex flex-col h-full border border-gray-700 overflow-hidden">
      <div className="p-4 bg-gray-700/50 border-b border-gray-700 flex-shrink-0">
        <h2 className="text-lg font-semibold text-white">Analysis & Output</h2>
      </div>
      <div className="flex-grow overflow-hidden">
        {renderContent()}
      </div>
    </div>
  );
};

export default ResultDisplay;
