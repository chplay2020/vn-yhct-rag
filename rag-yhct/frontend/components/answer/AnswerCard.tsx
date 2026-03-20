import { AskResponse } from "@/types/rag";

type AnswerCardProps = {
  response: AskResponse;
};

function CitationAwareText({ text }: { text: string }) {
  const parts = text.split(/(\[E\d+\])/g);
  return (
    <p className="whitespace-pre-wrap text-sm leading-7 text-slate-800">
      {parts.map((part, index) => {
        if (/^\[E\d+\]$/.test(part)) {
          return (
            <span
              key={`${part}-${index}`}
              className="mx-0.5 rounded bg-amber-100 px-1.5 py-0.5 text-xs font-semibold text-amber-800"
            >
              {part}
            </span>
          );
        }
        return <span key={`${part}-${index}`}>{part}</span>;
      })}
    </p>
  );
}

export function AnswerCard({ response }: AnswerCardProps) {
  const gatePass = response.gate_result?.pass;

  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_8px_24px_rgba(15,23,42,0.04)] md:p-5">
      <div className="mb-3 flex flex-wrap gap-2">
        <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-semibold text-slate-600">
          mode: {response.mode}
        </span>
        <span
          className={`rounded-full px-3 py-1 text-xs font-semibold ${
            gatePass === true
              ? "border border-emerald-200 bg-emerald-50 text-emerald-700"
              : gatePass === false
                ? "border border-orange-200 bg-orange-50 text-orange-700"
                : "border border-slate-200 bg-slate-50 text-slate-500"
          }`}
        >
          Gate: {gatePass === true ? "PASS" : gatePass === false ? "FAIL" : "N/A"}
        </span>
        <span
          className={`rounded-full px-3 py-1 text-xs font-semibold ${
            response.abstained
              ? "border border-orange-200 bg-orange-50 text-orange-700"
              : "border border-emerald-200 bg-emerald-50 text-emerald-700"
          }`}
        >
          {response.abstained ? "Abstained" : "Answered"}
        </span>
      </div>

      <h3 className="mb-2 text-lg font-semibold text-slate-900">Answer</h3>
      {response.abstained ? (
        <div className="mb-3 rounded-xl border border-orange-200 bg-orange-50 p-3 text-sm text-orange-700">
          Not enough evidence in the current documents.
        </div>
      ) : null}
      <CitationAwareText text={response.answer || "No answer generated."} />

      <h4 className="mb-2 mt-5 text-sm font-semibold uppercase tracking-wide text-slate-700">Key concepts</h4>
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
        <p className="text-sm text-slate-500">No key concepts returned.</p>
      )}

      <h4 className="mb-1 mt-5 text-sm font-semibold uppercase tracking-wide text-slate-700">Limits</h4>
      <p className="text-sm text-slate-700">{response.limits || "No explicit limits returned."}</p>

      <h4 className="mb-1 mt-5 text-sm font-semibold uppercase tracking-wide text-slate-700">Safety note</h4>
      <p className="rounded-lg border border-slate-200 bg-slate-50 p-2 text-sm text-slate-700">{response.safety_note}</p>
    </article>
  );
}
