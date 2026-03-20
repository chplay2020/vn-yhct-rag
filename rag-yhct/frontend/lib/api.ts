import { AskResponse, QueryControls } from "@/types/rag";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function askRag(query: string, controls: QueryControls): Promise<AskResponse> {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      mode: controls.mode,
      use_gate: controls.useGate,
      build_context: controls.buildContext,
      generate_answer: controls.generateAnswer,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API error ${response.status}: ${text}`);
  }

  return (await response.json()) as AskResponse;
}

export function parseCitations(answer: string): string[] {
  const matches = answer.match(/\[E\d+\]/g) ?? [];
  return Array.from(new Set(matches.map((m) => m.replace("[", "").replace("]", ""))));
}
