import { AskResponse, QueryControls } from "@/types/rag";

// Base URL của backend RAG, ưu tiên biến môi trường khi deploy.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function askRag(query: string, controls: QueryControls): Promise<AskResponse> {
  // Gửi câu hỏi và các tham số điều khiển truy vấn lên endpoint /api/ask.
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

  // Nếu backend trả lỗi, lấy raw text để dễ debug hơn thay vì chỉ status code.
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API error ${response.status}: ${text}`);
  }

  // Parse JSON và ép kiểu về AskResponse cho TypeScript.
  return (await response.json()) as AskResponse;
}

export function parseCitations(answer: string): string[] {
  // Bắt tất cả citation dạng [E123] trong câu trả lời.
  const matches = answer.match(/\[E\d+\]/g) ?? [];
  // Loại dấu [] và khử trùng lặp, trả về mảng như ["E1", "E2"].
  return Array.from(new Set(matches.map((m) => m.replace("[", "").replace("]", ""))));
}
