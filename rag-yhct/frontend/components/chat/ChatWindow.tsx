import { ChatTurn } from "@/types/rag";
import { AnswerCard } from "@/components/answer/AnswerCard";
import { MessageBubble } from "@/components/chat/MessageBubble";

type ChatWindowProps = {
  turns: ChatTurn[];
  loading: boolean;
  pendingQuery: string;
  onSelectTurn: (turnId: string) => void;
  selectedTurnId: string | null;
  onCitationClick?: (citationId: string) => void;
};

export function ChatWindow({
  turns,
  loading,
  pendingQuery,
  onSelectTurn,
  selectedTurnId,
  onCitationClick,
}: ChatWindowProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_8px_24px_rgba(15,23,42,0.04)] md:p-5">
      <h2 className="mb-3 text-lg font-semibold text-slate-900">Hội thoại</h2>
      <div className="space-y-4">
        {turns.map((turn) => {
          const isSelected = selectedTurnId === turn.id;
          return (
            <div
              key={turn.id}
              className={`rounded-xl border p-3 ${isSelected ? "border-blue-300 bg-blue-50/40" : "border-transparent"}`}
            >
              <button
                type="button"
                onClick={() => onSelectTurn(turn.id)}
                className="mb-2 text-xs font-medium text-slate-500 underline-offset-2 hover:underline"
              >
                Lượt {turn.id.slice(-4)}
              </button>
              <MessageBubble role="user" text={turn.query} />
              <div className="mt-3">
                {turn.error ? (
                  <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">{turn.error}</div>
                ) : turn.response ? (
                  <AnswerCard response={turn.response} onCitationClick={onCitationClick} />
                ) : (
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-500">
                    Đang chờ phản hồi...
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {loading && pendingQuery ? (
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-600">
            Đang chạy pipeline cho: <span className="font-medium">{pendingQuery}</span>
          </div>
        ) : null}

        {!turns.length ? (
          <p className="text-sm text-slate-500">Chưa có hội thoại. Hãy nhập câu hỏi đầu tiên ở bên dưới.</p>
        ) : null}
      </div>
    </section>
  );
}
