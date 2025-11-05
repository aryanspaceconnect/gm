import React from 'react';
import { FitnessData } from '../services/geminiService';

interface FitnessChartProps {
  fitnessData: FitnessData[];
}

export const FitnessChart: React.FC<FitnessChartProps> = ({ fitnessData }) => {
  if (!fitnessData || fitnessData.length === 0) return null;

  const padding = 40;
  const width = 500;
  const height = 300;
  
  const maxX = Math.max(...fitnessData.map(d => d.generation));
  const maxY = 1; // Fitness (Accuracy) is bounded 0-1

  const getX = (gen: number) => padding + (gen / maxX) * (width - padding * 2);
  const getY = (fit: number) => height - padding - (fit / maxY) * (height - padding * 2);

  const path = fitnessData
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(d.generation)} ${getY(d.fitness)}`)
    .join(' ');

  return (
    <div>
        <h3 className="text-lg font-semibold text-cyan-400 border-b border-gray-700 pb-2 mb-3">
            Accuracy Over Generations
        </h3>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-gray-900/50 rounded-md">
            {/* Y Axis */}
            <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#4a5568" />
            {[0, 0.25, 0.5, 0.75, 1.0].map(val => (
                <g key={val}>
                    <text x={padding - 10} y={getY(val) + 5} fill="#cbd5e0" textAnchor="end" fontSize="10">{val.toFixed(2)}</text>
                    <line x1={padding} y1={getY(val)} x2={width-padding} y2={getY(val)} stroke="#4a5568" strokeDasharray="2,2" />
                </g>
            ))}
             <text transform={`translate(${padding/4}, ${height/2}) rotate(-90)`} fill="#cbd5e0" textAnchor="middle" fontSize="12">Accuracy</text>

            {/* X Axis */}
            <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#4a5568" />
            {fitnessData.map(d => (
                <g key={d.generation}>
                    <text x={getX(d.generation)} y={height-padding+15} fill="#cbd5e0" textAnchor="middle" fontSize="10">{d.generation}</text>
                </g>
            ))}
            <text x={width/2} y={height-padding+35} fill="#cbd5e0" textAnchor="middle" fontSize="12">Generation</text>


            {/* Line */}
            <path d={path} fill="none" stroke="#06b6d4" strokeWidth="2" />
             {fitnessData.map(d => (
                <circle key={d.generation} cx={getX(d.generation)} cy={getY(d.fitness)} r="3" fill="#06b6d4" />
            ))}
        </svg>
    </div>
  );
};

interface MetricsDisplayProps {
    metrics: { score: number; accuracy: number; novelty: number };
}

export const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ metrics }) => {
    const metricItems = [
        { label: 'Overall Score', value: metrics.score.toFixed(4), unit: '' },
        { label: 'Final Accuracy', value: metrics.accuracy.toFixed(4), unit: '' },
        { label: 'Final Novelty', value: metrics.novelty.toFixed(4), unit: '' },
    ];

    return (
         <div>
            <h3 className="text-lg font-semibold text-cyan-400 border-b border-gray-700 pb-2 mb-3">
                Final Evolved Metrics
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {metricItems.map(item => (
                    <div key={item.label} className="bg-gray-900/50 p-4 rounded-lg text-center">
                        <p className="text-sm text-gray-400">{item.label}</p>
                        <p className="text-2xl font-bold text-white">
                            {item.value}
                            <span className="text-base font-normal text-gray-500 ml-1">{item.unit}</span>
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
}