import { EvidenceItem } from "@/types/rag";
import { RefObject } from "react";

type EvidenceCardProps = {
  item: EvidenceItem;
  index: number;
  highlighted: boolean;
  cardRef?: RefObject<HTMLElement>;
  flash: boolean;
};

export function EvidenceCard({ item, index, highlighted, cardRef, flash }: EvidenceCardProps) {
  const citationId = String(item.citation_id ?? `E${index + 1}`);
  return (
    <article
      ref={cardRef}
      className={`rounded-xl border p-3 ${
        flash
          ? "border-amber-400 bg-amber-100"
          : highlighted
            ? "border-amber-300 bg-amber-50"
            : "border-slate-200 bg-slate-50"
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <strong className="text-sm text-slate-900">{citationId}</strong>
          {highlighted ? (
            <span className="rounded-full border border-amber-300 bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-800">
              Dùng trong trả lời
            </span>
          ) : null}
        </div>
        <span className="text-xs text-slate-500">Điểm: {String(item.score ?? "N/A")}</span>
      </div>

      <p className="my-2 text-sm text-slate-700">{String(item.snippet ?? "")}</p>

      <dl className="grid grid-cols-[max-content_1fr] gap-x-2 gap-y-1 text-xs">
        <dt className="text-slate-500">Tiêu đề</dt>
        <dd className="truncate text-slate-700">{String(item.title ?? "N/A")}</dd>
        <dt className="text-slate-500">Mục</dt>
        <dd className="truncate text-slate-700">{String(item.section_heading ?? "N/A")}</dd>
        <dt className="text-slate-500">Trang</dt>
        <dd className="truncate text-slate-700">{String(item.page_range ?? "N/A")}</dd>
      </dl>

      <details className="mt-2">
        <summary className="cursor-pointer text-xs font-medium text-slate-600">Chi tiết kỹ thuật</summary>
        <dl className="mt-2 grid grid-cols-[max-content_1fr] gap-x-2 gap-y-1 text-xs">
          <dt className="text-slate-500">chunk_id</dt>
          <dd className="truncate text-slate-700">{String(item.chunk_id ?? "N/A")}</dd>
          <dt className="text-slate-500">parent_id</dt>
          <dd className="truncate text-slate-700">{String(item.parent_id ?? "N/A")}</dd>
          <dt className="text-slate-500">file_path</dt>
          <dd className="truncate text-slate-700">{String(item.file_path ?? "N/A")}</dd>
        </dl>
      </details>
    </article>
  );
}
