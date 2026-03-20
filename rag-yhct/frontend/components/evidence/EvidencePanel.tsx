import { EvidenceCard } from "@/components/evidence/EvidenceCard";
import { EvidenceItem } from "@/types/rag";

type EvidencePanelProps = {
  evidence: EvidenceItem[];
  citedIds: string[];
};

export function EvidencePanel({ evidence, citedIds }: EvidencePanelProps) {
  const citedSet = new Set(citedIds);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_8px_24px_rgba(15,23,42,0.04)] md:p-5">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-900">Evidence</h3>
        <span className="text-xs text-slate-500">{evidence.length} items</span>
      </div>

      {evidence.length ? (
        <div className="grid max-h-[42vh] gap-3 overflow-auto pr-1 lg:max-h-[55vh]">
          {evidence.map((item, index) => {
            const citationId = String(item.citation_id ?? `E${index + 1}`);
            return (
              <EvidenceCard
                key={`${citationId}-${index}`}
                item={item}
                index={index}
                highlighted={citedSet.has(citationId)}
              />
            );
          })}
        </div>
      ) : (
        <p className="text-sm text-slate-500">No evidence returned.</p>
      )}
    </section>
  );
}
