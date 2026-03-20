export type RetrievalMode = "vector" | "bm25" | "hybrid_rrf";

export type QueryControls = {
  mode: RetrievalMode;
  useGate: boolean;
  buildContext: boolean;
  generateAnswer: boolean;
};

export type EvidenceItem = {
  citation_id?: string;
  snippet?: string;
  chunk_id?: string;
  parent_id?: string;
  score?: number | string;
  title?: string;
  page_range?: string;
  section_heading?: string;
  file_path?: string;
  [key: string]: unknown;
};

export type GateFeatures = {
  top1_score?: number;
  top2_score?: number;
  top1_top2_gap?: number;
  evidence_count?: number;
  distinct_parent_count?: number;
  distinct_source_count?: number;
  [key: string]: unknown;
};

export type GateResult = {
  pass?: boolean;
  reason?: string;
  predicted_citation_count?: number;
  gate_features?: GateFeatures;
  [key: string]: unknown;
};

export type AskResponse = {
  query: string;
  mode: string;
  answer: string;
  key_concepts: string[];
  limits: string;
  safety_note: string;
  abstained: boolean;
  gate_result: GateResult | null;
  evidence: EvidenceItem[];
  retrieval_results: Record<string, unknown>[];
  context_debug: Record<string, unknown>;
};

export type ChatTurn = {
  id: string;
  query: string;
  controls: QueryControls;
  response?: AskResponse;
  error?: string;
};
