import type { QueryResponse } from "@/lib/api";

export interface ChatMessage {
	id: number;
	role: "user" | "assistant";
	content: string;
	context?: QueryResponse["context"];
	error?: string;
	eval?: Record<string, number>;
	eval_job_id?: string;
	eval_status?: "pending" | "complete" | "error";
}
