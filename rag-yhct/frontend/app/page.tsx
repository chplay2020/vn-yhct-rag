"use client";

import { FormEvent, useMemo, useState } from "react";

import { ChatWindow } from "@/components/chat/ChatWindow";
import { AdvancedDebugPanel } from "@/components/debug/AdvancedDebugPanel";
import { EvidencePanel } from "@/components/evidence/EvidencePanel";
import { GateStatusPanel } from "@/components/status/GateStatusPanel";
import { askRag, parseCitations } from "@/lib/api";
import { AskResponse, ChatTurn, QueryControls, RetrievalMode } from "@/types/rag";

const EXAMPLE_QUERIES = [
  "tác dụng của cây ngải cứu",
  "tác dụng của cây sả",
  "thuốc nào chữa khỏi hoàn toàn mọi loại ung thư",
];

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [controls, setControls] = useState<QueryControls>({
    mode: "hybrid_rrf",
    useGate: true,
    buildContext: true,
    generateAnswer: true,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [pendingQuery, setPendingQuery] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null);
  const [activeCitationId, setActiveCitationId] = useState<string | null>(null);

  const activeTurn = useMemo(() => {
    if (!turns.length) return null;
    if (!selectedTurnId) return turns[turns.length - 1] ?? null;
    return turns.find((turn) => turn.id === selectedTurnId) ?? turns[turns.length - 1] ?? null;
  }, [selectedTurnId, turns]);

  const activeResponse: AskResponse | null = activeTurn?.response ?? null;
  const citedIds = useMemo(() => parseCitations(activeResponse?.answer ?? ""), [activeResponse?.answer]);

  function updateControl<K extends keyof QueryControls>(key: K, value: QueryControls[K]) {
    setControls((prev) => ({ ...prev, [key]: value }));
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || isLoading) {
      return;
    }

    const turnId = `${Date.now()}`;
    const currentControls = { ...controls };
    setPendingQuery(trimmed);
    setTurns((prev) => [...prev, { id: turnId, query: trimmed, controls: currentControls }]);
    setSelectedTurnId(turnId);
    setIsLoading(true);

    try {
      const payload = await askRag(trimmed, currentControls);
      setTurns((prev) =>
        prev.map((turn) => (turn.id === turnId ? { ...turn, response: payload } : turn)),
      );
      setQuery("");
      setActiveCitationId(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Lỗi không xác định";
      setTurns((prev) => prev.map((turn) => (turn.id === turnId ? { ...turn, error: message } : turn)));
    } finally {
      setPendingQuery("");
      setIsLoading(false);
    }
  }

  return (
    <main className="mx-auto w-[min(1320px,calc(100%-2rem))] py-8 md:py-12">
      <section className="mb-5 space-y-2">
        <p className="m-0 text-xs font-bold uppercase tracking-[0.08em] text-blue-700">RAG YHCT Thesis Demo</p>
        <h1 className="text-2xl font-semibold leading-tight text-slate-900 md:text-3xl">Không gian Hội thoại + Bằng chứng</h1>
        <p className="m-0 text-sm text-slate-600 md:text-base">
          Hybrid RRF → Gate → Ngữ cảnh tập trung → Local LLM + Trích dẫn. Giao diện ưu tiên khả năng truy vết bằng
          chứng, trạng thái tin cậy và hiển thị an toàn cho demo luận văn.
        </p>
      </section>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(360px,1fr)]">
        <div className="space-y-4">
          <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_8px_24px_rgba(15,23,42,0.04)] md:p-5">
            <div className="mb-3 grid grid-cols-1 gap-2 md:grid-cols-2 lg:grid-cols-4">
              <label className="flex flex-col gap-1 text-xs font-medium text-slate-600">
                Chế độ truy xuất
                <select
                  value={controls.mode}
                  onChange={(e) => updateControl("mode", e.target.value as RetrievalMode)}
                  className="rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm text-slate-800"
                >
                  <option value="vector">vector</option>
                  <option value="bm25">bm25</option>
                  <option value="hybrid_rrf">hybrid_rrf</option>
                </select>
              </label>

              <label className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={controls.useGate}
                  onChange={(e) => updateControl("useGate", e.target.checked)}
                />
                Dùng Gate
              </label>

              <label className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={controls.buildContext}
                  onChange={(e) => updateControl("buildContext", e.target.checked)}
                />
                Tạo ngữ cảnh
              </label>

              <label className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={controls.generateAnswer}
                  onChange={(e) => updateControl("generateAnswer", e.target.checked)}
                />
                Sinh câu trả lời
              </label>
            </div>

            <div className="mb-3 flex flex-wrap gap-2">
              {EXAMPLE_QUERIES.map((sample) => (
                <button
                  key={sample}
                  type="button"
                  onClick={() => setQuery(sample)}
                  className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-700 hover:bg-slate-100"
                >
                  {sample}
                </button>
              ))}
            </div>

            <form onSubmit={onSubmit} className="flex gap-2">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Nhập câu hỏi tiếp theo..."
                className="min-w-0 flex-1 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
              />
              <button
                type="submit"
                disabled={isLoading || !query.trim()}
                className="rounded-xl bg-blue-700 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isLoading ? "Đang chạy..." : "Gửi"}
              </button>
            </form>
          </section>

          <ChatWindow
            turns={turns}
            loading={isLoading}
            pendingQuery={pendingQuery}
            onSelectTurn={setSelectedTurnId}
            selectedTurnId={selectedTurnId}
            onCitationClick={setActiveCitationId}
          />
        </div>

        <aside className="space-y-4 xl:sticky xl:top-4 xl:self-start">
          <EvidencePanel
            evidence={activeResponse?.evidence ?? []}
            citedIds={citedIds}
            activeCitationId={activeCitationId}
          />
          <GateStatusPanel response={activeResponse} controls={activeTurn?.controls ?? controls} />
          <AdvancedDebugPanel response={activeResponse} />
        </aside>
      </section>
    </main>
  );
}
