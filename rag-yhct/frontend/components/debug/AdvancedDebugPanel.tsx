import { AskResponse } from "@/types/rag";

type AdvancedDebugPanelProps = {
  response: AskResponse | null;
};

export function AdvancedDebugPanel({ response }: AdvancedDebugPanelProps) {
  const selectedParentIds = response?.context_debug?.selected_parent_ids ?? [];
  const filteredOutParents = response?.context_debug?.filtered_out_parents ?? [];
  const finalAnswerChunkIds = response?.context_debug?.final_answer_chunk_ids ?? [];

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_8px_24px_rgba(15,23,42,0.04)] md:p-5">
      <h3 className="mb-3 text-base font-semibold text-slate-900">Advanced Debug</h3>
      <details>
        <summary className="cursor-pointer text-sm font-medium text-slate-700">selected_parent_ids</summary>
        <pre className="mt-2 max-h-44 overflow-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-200">{JSON.stringify(selectedParentIds, null, 2)}</pre>
      </details>
      <details className="mt-2">
        <summary className="cursor-pointer text-sm font-medium text-slate-700">filtered_out_parents</summary>
        <pre className="mt-2 max-h-44 overflow-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-200">{JSON.stringify(filteredOutParents, null, 2)}</pre>
      </details>
      <details className="mt-2">
        <summary className="cursor-pointer text-sm font-medium text-slate-700">final_answer_chunk_ids</summary>
        <pre className="mt-2 max-h-44 overflow-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-200">{JSON.stringify(finalAnswerChunkIds, null, 2)}</pre>
      </details>
      <details className="mt-2">
        <summary className="cursor-pointer text-sm font-medium text-slate-700">raw evidence JSON</summary>
        <pre className="mt-2 max-h-56 overflow-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-200">
          {JSON.stringify(response?.evidence ?? [], null, 2)}
        </pre>
      </details>
      <details className="mt-2">
        <summary className="cursor-pointer text-sm font-medium text-slate-700">raw pipeline debug JSON</summary>
        <pre className="mt-2 max-h-56 overflow-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-200">
          {JSON.stringify(
            {
              gate_result: response?.gate_result ?? null,
              context_debug: response?.context_debug ?? {},
              retrieval_results_preview: (response?.retrieval_results ?? []).slice(0, 8),
            },
            null,
            2,
          )}
        </pre>
      </details>
    </section>
  );
}
