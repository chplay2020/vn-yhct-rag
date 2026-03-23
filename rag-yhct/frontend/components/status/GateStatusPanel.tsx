import { AskResponse, QueryControls } from "@/types/rag";

type GateStatusPanelProps = {
  response: AskResponse | null;
  controls: QueryControls;
};

function StatusChip({ label, tone }: { label: string; tone: "ok" | "warn" | "muted" }) {
  const cls =
    tone === "ok"
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : tone === "warn"
        ? "border-red-200 bg-red-50 text-red-700"
        : "border-slate-200 bg-slate-50 text-slate-600";
  return <span className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${cls}`}>{label}</span>;
}

export function GateStatusPanel({ response, controls }: GateStatusPanelProps) {
  const gate = response?.gate_result;
  const gatePass = gate?.pass;
  const gateFeatures = gate?.gate_features ?? {};
  const selectedParents = Array.isArray(response?.context_debug?.selected_parent_ids)
    ? (response?.context_debug?.selected_parent_ids as unknown[])
    : [];
  const tokensUsed =
    typeof response?.context_debug?.tokens_used === "number"
      ? (response?.context_debug?.tokens_used as number)
      : "N/A";

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_8px_24px_rgba(15,23,42,0.04)] md:p-5">
      <h3 className="mb-3 text-base font-semibold text-slate-900">Trạng thái tin cậy</h3>

      <div className="mb-3 flex flex-wrap gap-2">
        <StatusChip label={`Chế độ: ${response?.mode ?? controls.mode}`} tone="muted" />
        <StatusChip
          label={`Gate: ${gatePass === true ? "PASS" : gatePass === false ? "FAIL" : "N/A"}`}
          tone={gatePass === true ? "ok" : gatePass === false ? "warn" : "muted"}
        />
        <StatusChip
          label={`Từ chối: ${response?.abstained ? "Có" : "Không"}`}
          tone={response?.abstained ? "warn" : "ok"}
        />
      </div>

      <dl className="grid grid-cols-[max-content_1fr] gap-x-2 gap-y-1 text-xs">
        <dt className="text-slate-500">Số trích dẫn dự đoán</dt>
        <dd className="text-slate-700">{String(gate?.predicted_citation_count ?? "N/A")}</dd>
        <dt className="text-slate-500">Đã tạo ngữ cảnh</dt>
        <dd className="text-slate-700">{selectedParents.length > 0 ? "Có" : controls.buildContext ? "Có yêu cầu" : "Không"}</dd>
        <dt className="text-slate-500">Số token ngữ cảnh cuối</dt>
        <dd className="text-slate-700">{String(tokensUsed)}</dd>
        <dt className="text-slate-500">Đã sinh câu trả lời</dt>
        <dd className="text-slate-700">{response?.answer ? "Có" : controls.generateAnswer ? "Có yêu cầu" : "Không"}</dd>
      </dl>

      {gate ? (
        <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
          <p className="text-xs font-semibold text-slate-700">Chi tiết Gate</p>
          <p className="mt-1 text-xs text-slate-600">Lý do: {String(gate.reason ?? "N/A")}</p>
          <p className="mt-1 text-xs text-slate-600">
            top1 / top2 / gap: {String(gateFeatures.top1_score ?? "N/A")} / {String(gateFeatures.top2_score ?? "N/A")} /
            {" "}
            {String(gateFeatures.top1_top2_gap ?? "N/A")}
          </p>
          <p className="mt-1 text-xs text-slate-600">
            bằng chứng / parent / nguồn: {String(gateFeatures.evidence_count ?? "N/A")} /
            {" "}
            {String(gateFeatures.distinct_parent_count ?? "N/A")} / {String(gateFeatures.distinct_source_count ?? "N/A")}
          </p>
        </div>
      ) : null}
    </section>
  );
}
