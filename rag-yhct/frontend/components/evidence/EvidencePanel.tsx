"use client";

import { createRef, useEffect, useMemo, useRef, useState } from "react";

import { EvidenceCard } from "@/components/evidence/EvidenceCard";
import { EvidenceItem } from "@/types/rag";

type EvidencePanelProps = {
  evidence: EvidenceItem[];
  citedIds: string[];
  activeCitationId?: string | null;
};

export function EvidencePanel({ evidence, citedIds, activeCitationId }: EvidencePanelProps) {
  const citedSet = new Set(citedIds);
  const [flashCitationId, setFlashCitationId] = useState<string | null>(null);
  const refs = useRef<Record<string, ReturnType<typeof createRef<HTMLElement>>>>({});

  const sortedEvidence = useMemo(() => {
    return [...evidence].sort((a, b) => {
      const aId = String(a.citation_id ?? "");
      const bId = String(b.citation_id ?? "");
      const aUsed = citedSet.has(aId);
      const bUsed = citedSet.has(bId);
      if (aUsed === bUsed) return 0;
      return aUsed ? -1 : 1;
    });
  }, [evidence, citedIds]);

  useEffect(() => {
    if (!activeCitationId) return;
    if (!refs.current[activeCitationId]) {
      refs.current[activeCitationId] = createRef<HTMLElement>();
    }
    const target = refs.current[activeCitationId]?.current;
    if (!target) return;

    target.scrollIntoView({ behavior: "smooth", block: "center" });
    setFlashCitationId(activeCitationId);
    const timer = window.setTimeout(() => setFlashCitationId(null), 1400);
    return () => window.clearTimeout(timer);
  }, [activeCitationId]);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_8px_24px_rgba(15,23,42,0.04)] md:p-5">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-900">Bằng chứng</h3>
        <span className="text-xs text-slate-500">{sortedEvidence.length} mục</span>
      </div>

      {sortedEvidence.length ? (
        <div className="grid max-h-[42vh] gap-3 overflow-auto pr-1 lg:max-h-[55vh]">
          {sortedEvidence.map((item, index) => {
            const citationId = String(item.citation_id ?? `E${index + 1}`);
            if (!refs.current[citationId]) {
              refs.current[citationId] = createRef<HTMLElement>();
            }
            return (
              <EvidenceCard
                key={`${citationId}-${index}`}
                item={item}
                index={index}
                highlighted={citedSet.has(citationId)}
                cardRef={refs.current[citationId]}
                flash={flashCitationId === citationId}
              />
            );
          })}
        </div>
      ) : (
        <p className="text-sm text-slate-500">Không có bằng chứng trả về.</p>
      )}
    </section>
  );
}
