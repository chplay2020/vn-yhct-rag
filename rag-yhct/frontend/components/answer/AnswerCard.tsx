import { AskResponse } from "@/types/rag";

type AnswerCardProps = {
  response: AskResponse;
  onCitationClick?: (citationId: string) => void;
};

function CitationAwareText({
  text,
  onCitationClick,
}: {
  text: string;
  onCitationClick?: (citationId: string) => void;
}) {
  const parts = text.split(/(\[E\d+\])/g);
  return (
    <p className="whitespace-pre-wrap text-[15px] leading-7 text-slate-800">
      {parts.map((part, index) => {
        if (/^\[E\d+\]$/.test(part)) {
          const citationId = part.replace("[", "").replace("]", "");
          return (
            <button
              type="button"
              key={`${part}-${index}`}
              onClick={() => onCitationClick?.(citationId)}
              className="mx-0.5 rounded bg-amber-100 px-1.5 py-0.5 text-xs font-semibold text-amber-800 transition hover:bg-amber-200"
              title={`Xem bằng chứng ${citationId}`}
            >
              {part}
            </button>
          );
        }
        return <span key={`${part}-${index}`}>{part}</span>;
      })}
    </p>
  );
}

export function AnswerCard({ response, onCitationClick }: AnswerCardProps) {
  const gatePass = response.gate_result?.pass;
  const gateReason = String(response.gate_result?.reason ?? "").trim();

  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_8px_24px_rgba(15,23,42,0.04)] md:p-5">
      <div className="mb-3 flex flex-wrap gap-2">
        <span className="rounded-full border border-slate-300 bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">
          Chế độ: {response.mode}
        </span>
        <span
          className={`rounded-full px-3 py-1 text-xs font-semibold ${
            gatePass === true
              ? "border border-emerald-200 bg-emerald-50 text-emerald-700"
              : gatePass === false
                ? "border border-red-200 bg-red-50 text-red-700"
                : "border border-slate-200 bg-slate-50 text-slate-500"
          }`}
        >
          Gate: {gatePass === true ? "PASS" : gatePass === false ? "FAIL" : "N/A"}
        </span>
        <span
          className={`rounded-full px-3 py-1 text-xs font-semibold ${
            response.abstained
              ? "border border-amber-200 bg-amber-50 text-amber-700"
              : "border border-emerald-200 bg-emerald-50 text-emerald-700"
          }`}
        >
          {response.abstained ? "Từ chối trả lời" : "Đã trả lời"}
        </span>
      </div>

      <h3 className="mb-2 text-xl font-semibold text-slate-900">Câu trả lời</h3>
      {response.abstained ? (
        <div className="mb-3 rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800">
          <p className="font-semibold">Không đủ căn cứ trong tài liệu hiện có.</p>
          {gateReason ? <p className="mt-1">Lý do gate: {gateReason}</p> : null}
        </div>
      ) : null}
      <CitationAwareText
        text={response.answer || "Hiện chưa có nội dung trả lời."}
        onCitationClick={onCitationClick}
      />

      <h4 className="mb-2 mt-6 text-sm font-semibold uppercase tracking-wide text-slate-700">Khái niệm chính</h4>
      {response.key_concepts.length ? (
        <ul className="m-0 flex list-none flex-wrap gap-2 p-0">
          {response.key_concepts.map((concept) => (
            <li
              key={concept}
              className="rounded-full border border-indigo-200 bg-indigo-50 px-2.5 py-1 text-xs font-medium text-indigo-700"
            >
              {concept}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-slate-500">Không có khái niệm chính.</p>
      )}

      <h4 className="mb-1 mt-5 text-sm font-semibold uppercase tracking-wide text-slate-700">Giới hạn</h4>
      <p className="text-sm text-slate-700">{response.limits || "Không có mô tả giới hạn."}</p>

      <h4 className="mb-1 mt-5 text-sm font-semibold uppercase tracking-wide text-slate-700">Lưu ý an toàn</h4>
      <p className="rounded-lg border border-slate-200 bg-slate-50 p-2 text-sm text-slate-700">{response.safety_note}</p>
    </article>
  );
}
