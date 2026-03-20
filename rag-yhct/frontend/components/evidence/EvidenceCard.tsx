import { EvidenceItem } from "@/types/rag";

type EvidenceCardProps = {
  item: EvidenceItem;
  index: number;
  highlighted: boolean;
};

export function EvidenceCard({ item, index, highlighted }: EvidenceCardProps) {
  const citationId = String(item.citation_id ?? `E${index + 1}`);
  return (
    <article
      className={`rounded-xl border p-3 ${
        highlighted ? "border-amber-300 bg-amber-50" : "border-slate-200 bg-slate-50"
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <strong className="text-sm text-slate-900">{citationId}</strong>
        <span className="text-xs text-slate-500">score: {String(item.score ?? "N/A")}</span>
      </div>

      <p className="my-2 text-sm text-slate-700">{String(item.snippet ?? "")}</p>

      <dl className="grid grid-cols-[max-content_1fr] gap-x-2 gap-y-1 text-xs">
        <dt className="text-slate-500">chunk_id</dt>
        <dd className="truncate text-slate-700">{String(item.chunk_id ?? "N/A")}</dd>
        <dt className="text-slate-500">parent_id</dt>
        <dd className="truncate text-slate-700">{String(item.parent_id ?? "N/A")}</dd>
        <dt className="text-slate-500">title</dt>
        <dd className="truncate text-slate-700">{String(item.title ?? "N/A")}</dd>
        <dt className="text-slate-500">page_range</dt>
        <dd className="truncate text-slate-700">{String(item.page_range ?? "N/A")}</dd>
        <dt className="text-slate-500">section_heading</dt>
        <dd className="truncate text-slate-700">{String(item.section_heading ?? "N/A")}</dd>
        <dt className="text-slate-500">file_path</dt>
        <dd className="truncate text-slate-700">{String(item.file_path ?? "N/A")}</dd>
      </dl>
    </article>
  );
}
