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
    score: number;
    accuracy: number;
    novelty: number;
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
    As an expert AI and physics simulation analyst, your task is to provide a comprehensive, multi-faceted review of the provided Python script and its mocked execution output. The script uses genetic programming and novelty search to evolve a 2D physics engine for a bouncing particle.

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
        *   Provide a narrative analysis formatted in Markdown. Focus on the **implications and outcomes**.
        *   **Code Purpose & Methodology:** Explain the goal of evolving a physics engine. Describe the modular approach using 'BLOCKS' (gravity, air_drag, euler, rk4, bounce) and how the \`Individual\` class combines them. Contrast this with a monolithic script.
        *   **Execution Outcome Interpretation:** Analyze the evolutionary process based on the logs. The 'Best score' is a composite of accuracy, efficiency, and novelty. Explain why 'Accuracy' rises while 'Novelty' might fluctuate. Interpret the early stopping condition.
        *   **Implications & Conclusion:** What does this experiment show about evolving code with specific, functional modules? Discuss the trade-offs in the final score's components.

    2.  **fitnessChartData (array of objects):**
        *   Extract the 'Gen' and 'Accuracy' values from each log entry in the mocked output.
        *   Format this as an array of objects: \`{ "generation": <Gen_Number>, "fitness": <Accuracy_Value> }\`. Note: Use the 'Accuracy' value for the 'fitness' key.
        *   Example: \`[ { "generation": 1, "fitness": 0.1532 }, { "generation": 20, "fitness": 0.6789 }, ... ]\`

    3.  **finalMetrics (object):**
        *   From the LAST log entry, extract the final 'Best score', 'Accuracy', and 'Novelty' values.
        *   Format this as an object: \`{ "score": <value>, "accuracy": <value>, "novelty": <value> }\`.

    4.  **trajectoryData (object):**
        *   Based on the physics described in the Python script (initial state [x=0, y=1, vx=1, vy=0], gravity, ground at y=0 with restitution=0.8), generate plausible trajectory data.
        *   The object should contain two keys: "truth" and "evolved".
        *   Each key should have an array of 100 coordinate objects \`{ "x": <value>, "y": <value> }\`, corresponding to the \`STEPS = 100\` constant.
        *   The "truth" path should be a perfect parabolic arc with bounces, based on the verlet integration.
        *   The "evolved" path should be very close to the "truth" path, reflecting the high final accuracy score (e.g., > 0.92). It might show very minor deviations.
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
            systemInstruction: "You are an expert AI and physics simulation analyst. The user has just seen your initial analysis of an evolved physics engine. Answer their follow-up questions concisely, referring to the original context of the Python script (modular blocks, genetic algorithm) and its output."
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