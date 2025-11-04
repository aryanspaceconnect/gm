import { GoogleGenAI, Chat } from "@google/genai";

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

// --- Type Definitions for Structured Data ---

export interface FitnessData {
  generation: number;
  fitness: number;
}

export interface TrajectoryPoint {
  x: number;
  y: number;
}

export interface AnalysisData {
  summary: string; // Markdown formatted text
  fitnessChartData: FitnessData[];
  finalMetrics: {
    fitness: number;
    novelty: number;
    runtime: number; // in ms
  };
  trajectoryData: {
    truth: TrajectoryPoint[];
    evolved: TrajectoryPoint[];
  };
}

let chat: Chat | null = null;

export async function analyzeCodeAndOutput(code: string, output: string): Promise<AnalysisData> {
  const model = "gemini-2.5-pro";

  const prompt = `
    As an expert AI and physics simulation analyst, your task is to provide a comprehensive, multi-faceted review of the provided Python script and its mocked execution output. The script uses genetic programming to evolve a physics engine.

    **Python Code:**
    \`\`\`python
    ${code}
    \`\`\`

    **Mocked Execution Output:**
    \`\`\`
    ${output}
    \`\`\`

    **Your Task:**
    Generate a JSON object containing a full analysis. The JSON object must have the following structure:
    {
      "summary": "...",
      "fitnessChartData": [ ... ],
      "finalMetrics": { ... },
      "trajectoryData": { ... }
    }

    **Detailed Instructions for each JSON field:**

    1.  **summary (string):**
        *   Provide a narrative analysis formatted in Markdown. Focus on the **implications and outcomes** more than just describing the code.
        *   **Code Purpose & Methodology:** Briefly explain the goal and the genetic programming approach.
        *   **Execution Outcome Interpretation:** Analyze the evolutionary process based on the logs. What does the progression of Fitness and Novelty tell us? How efficient is the final evolved engine?
        *   **Implications & Conclusion:** What does this experiment demonstrate about AI-driven algorithm discovery? What are the broader implications of evolving physics engines?

    2.  **fitnessChartData (array of objects):**
        *   Extract the 'Gen' and 'Best Fit' values from each log entry in the mocked output.
        *   Format this as an array of objects: \`{ "generation": <Gen_Number>, "fitness": <Best_Fit_Value> }\`.
        *   Example: \`[ { "generation": 1, "fitness": 0.0123 }, { "generation": 11, "fitness": 0.4582 }, ... ]\`

    3.  **finalMetrics (object):**
        *   From the LAST log entry (Gen 191), extract the final 'Best Fit', 'Nov', and 'RT' values.
        *   Format this as an object: \`{ "fitness": <value>, "novelty": <value>, "runtime": <value_in_ms> }\`. Convert the runtime from seconds to milliseconds.

    4.  **trajectoryData (object):**
        *   Based on the physics described in the Python script's comments (initial state [x=0, y=5, vx=2, vy=0], gravity, ground at y=0 with restitution), generate plausible trajectory data.
        *   The object should contain two keys: "truth" and "evolved".
        *   Each key should have an array of 200 coordinate objects \`{ "x": <value>, "y": <value> }\`.
        *   The "truth" path should be a perfect parabolic arc with bounces.
        *   The "evolved" path should be very close to the "truth" path, reflecting the high final fitness score of ~0.9998. It might have very slight deviations.
  `;

  try {
    const response = await ai.models.generateContent({
      model: model,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
      }
    });
    
    const analysisData = JSON.parse(response.text) as AnalysisData;

    // Initialize chat for follow-ups
    chat = ai.chats.create({
        model: 'gemini-2.5-flash', // Use a faster model for chat
        history: [
            { role: 'user', parts: [{ text: prompt }] },
            { role: 'model', parts: [{ text: response.text }] },
        ],
        config: {
            systemInstruction: "You are an expert AI and physics simulation analyst. The user has just seen your initial analysis. Answer their follow-up questions concisely, referring to the original context of the Python script and its output. When asked about code, provide specific, helpful examples."
        }
    });

    return analysisData;

  } catch (error) {
    console.error("Gemini API call failed:", error);
    if (error instanceof Error) {
        throw new Error(`Gemini API Error: ${error.message}`);
    }
    throw new Error("An unknown error occurred while communicating with the Gemini API.");
  }
}

export async function askFollowUp(question: string): Promise<string> {
    if (!chat) {
        throw new Error("Chat session not initialized. Please run a full analysis first.");
    }
    try {
        const response = await chat.sendMessage({ message: question });
        return response.text;
    } catch (error) {
        console.error("Gemini chat call failed:", error);
        if (error instanceof Error) {
            throw new Error(`Gemini API Error: ${error.message}`);
        }
        throw new Error("An unknown error occurred during the follow-up chat.");
    }
}
