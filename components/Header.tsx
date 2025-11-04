
import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="bg-gray-800 shadow-md">
      <div className="container mx-auto px-4 md:px-6 py-4 flex items-center gap-4">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-8 w-8 text-cyan-500"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
        </svg>
        <h1 className="text-xl md:text-2xl font-bold text-white tracking-wide">
          Python Physics Engine Analyzer
        </h1>
      </div>
    </header>
  );
};

export default Header;
