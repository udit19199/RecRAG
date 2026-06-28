import type { ChatMessage } from "@/features/chat/types";
import type { ProvidersResponse } from "@/lib/api";

export type SlotKey = "A" | "B";

export type SlotState = {
	vision: string | null;
	embedding: string | null;
	llm: string | null;
	hasDocuments: boolean;
	isChecking: boolean;
};

const INITIAL_SLOT_STATE: SlotState = {
	vision: null,
	embedding: null,
	llm: null,
	hasDocuments: false,
	isChecking: false,
};

export type ModelComparisonState = {
	providers: ProvidersResponse | null;
	error: string | null;
	slotA: SlotState;
	slotB: SlotState;
	ingestingSlot: SlotKey | null;
	isIngesting: boolean;
	messagesA: ChatMessage[];
	messagesB: ChatMessage[];
	isQuerying: boolean;
};

export const INITIAL_MODEL_COMPARISON_STATE: ModelComparisonState = {
	providers: null,
	error: null,
	slotA: INITIAL_SLOT_STATE,
	slotB: INITIAL_SLOT_STATE,
	ingestingSlot: null,
	isIngesting: false,
	messagesA: [],
	messagesB: [],
	isQuerying: false,
};

export type ModelComparisonAction =
	| { type: "set_providers"; providers: ProvidersResponse }
	| { type: "set_error"; error: string | null }
	| { type: "patch_slot"; slot: SlotKey; patch: Partial<SlotState> }
	| {
			type: "set_ingestion";
			isIngesting: boolean;
			ingestingSlot: SlotKey | null;
	  }
	| { type: "append_messages"; slot: SlotKey; messages: ChatMessage[] }
	| { type: "set_querying"; isQuerying: boolean };

export function modelComparisonReducer(
	state: ModelComparisonState,
	action: ModelComparisonAction,
): ModelComparisonState {
	switch (action.type) {
		case "set_providers":
			return { ...state, providers: action.providers };
		case "set_error":
			return { ...state, error: action.error };
		case "patch_slot": {
			const slotKey = action.slot === "A" ? "slotA" : "slotB";
			return {
				...state,
				[slotKey]: { ...state[slotKey], ...action.patch },
			};
		}
		case "set_ingestion":
			return {
				...state,
				isIngesting: action.isIngesting,
				ingestingSlot: action.ingestingSlot,
			};
		case "append_messages":
			return action.slot === "A"
				? {
						...state,
						messagesA: [...state.messagesA, ...action.messages],
					}
				: {
						...state,
						messagesB: [...state.messagesB, ...action.messages],
					};
		case "set_querying":
			return { ...state, isQuerying: action.isQuerying };
		default:
			return state;
	}
}
