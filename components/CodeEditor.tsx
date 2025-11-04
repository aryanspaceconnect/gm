
import React from 'react';
import { PlayIcon, ArrowPathIcon } from './Icons';

interface CodeEditorProps {
  code: string;
  setCode: (code: string) => void;
  onRun: () => void;
  isLoading: boolean;
}

const CodeEditor: React.FC<CodeEditorProps> = ({ code, setCode, onRun, isLoading }) => {
  return (
    <div className="bg-gray-800 rounded-lg shadow-lg flex flex-col h-full overflow-hidden border border-gray-700">
      <div className="p-4 bg-gray-700/50 border-b border-gray-700 flex justify-between items-center">
        <h2 className="text-lg font-semibold text-white">Python Code</h2>
        <span className="text-sm text-gray-400">physics_engine.py</span>
      </div>
      <textarea
        value={code}
        onChange={(e) => setCode(e.target.value)}
        className="flex-grow p-4 bg-transparent text-gray-300 font-mono text-sm w-full h-full resize-none focus:outline-none"
        spellCheck="false"
      />
      <div className="p-4 bg-gray-800 border-t border-gray-700">
        <button
          onClick={onRun}
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-cyan-600 text-white font-bold rounded-md hover:bg-cyan-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-cyan-500"
        >
          {isLoading ? (
            <>
              <ArrowPathIcon className="h-5 w-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <PlayIcon className="h-5 w-5" />
              Run & Analyze with Gemini
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default CodeEditor;
