import React, { useRef, useEffect } from 'react';
import { TrajectoryPoint } from '../services/geminiService';

interface SimulationCanvasProps {
  trajectoryData: {
    truth: TrajectoryPoint[];
    evolved: TrajectoryPoint[];
  };
}

const SimulationCanvas: React.FC<SimulationCanvasProps> = ({ trajectoryData }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !trajectoryData) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 30;

    const allPoints = [...trajectoryData.truth, ...trajectoryData.evolved];
    const xMin = Math.min(...allPoints.map(p => p.x));
    const xMax = Math.max(...allPoints.map(p => p.x));
    const yMin = 0; // Ground is at 0
    const yMax = Math.max(...allPoints.map(p => p.y));

    const scaleX = (width - 2 * padding) / (xMax - xMin);
    const scaleY = (height - 2 * padding) / (yMax - yMin);
    const scale = Math.min(scaleX, scaleY);
    
    const offsetX = padding - xMin * scale + (width - 2*padding - (xMax - xMin)*scale)/2;
    const offsetY = height - padding;

    const transformX = (x: number) => x * scale + offsetX;
    const transformY = (y: number) => offsetY - y * scale;

    let frame = 0;
    let animationFrameId: number;

    const drawPath = (points: TrajectoryPoint[], color: string, upToFrame?: number) => {
        ctx.beginPath();
        const pathPoints = upToFrame ? points.slice(0, upToFrame) : points;
        if (pathPoints.length === 0) return;

        ctx.moveTo(transformX(pathPoints[0].x), transformY(pathPoints[0].y));
        pathPoints.forEach(p => {
            ctx.lineTo(transformX(p.x), transformY(p.y));
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
    };

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw ground
      ctx.beginPath();
      ctx.moveTo(0, transformY(0));
      ctx.lineTo(width, transformY(0));
      ctx.strokeStyle = '#cbd5e0';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = '#cbd5e0';
      ctx.fillText('Ground', padding, transformY(0) + 15);
      
      // Draw paths
      drawPath(trajectoryData.truth, '#4a5568'); // Faded truth path
      drawPath(trajectoryData.evolved, '#0891b2', frame); // Animated evolved path
      
      // Draw particle
      if(frame > 0 && frame <= trajectoryData.evolved.length) {
          const particle = trajectoryData.evolved[frame - 1];
          ctx.beginPath();
          ctx.arc(transformX(particle.x), transformY(particle.y), 5, 0, 2 * Math.PI);
          ctx.fillStyle = '#06b6d4';
          ctx.fill();
      }

      // Legend
      ctx.fillStyle = '#4a5568';
      ctx.fillRect(width - padding - 80, padding - 10, 10, 2);
      ctx.fillText('Truth', width - padding - 65, padding);
      ctx.fillStyle = '#0891b2';
      ctx.fillRect(width - padding - 80, padding + 5, 10, 2);
      ctx.fillText('Evolved', width - padding - 65, padding + 15);

      frame = (frame + 1) % (trajectoryData.evolved.length + 1);
      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [trajectoryData]);

  return (
    <div className="p-4 h-full flex flex-col items-center justify-center">
         <h3 className="text-lg font-semibold text-cyan-400 pb-2 mb-3 self-start">
            Evolved Trajectory vs. Ground Truth
        </h3>
        <canvas ref={canvasRef} width="500" height="350" className="bg-gray-900/50 rounded-md w-full max-w-[500px] h-auto"></canvas>
    </div>
  );
};

export default SimulationCanvas;
