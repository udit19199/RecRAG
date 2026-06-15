import type { ComponentType } from "react";
import {
	IconArrowsLeftRight,
	IconClipboardList,
	IconMessageChatbot,
	IconReportAnalytics,
} from "@tabler/icons-react";

export type NavItem = {
	href: string;
	label: string;
	description: string;
	icon: ComponentType<{ className?: string }>;
	/** Match active route by prefix (e.g. /recommend?run_id=…) */
	matchPrefix?: boolean;
};

export type NavSection = {
	title: string;
	caption: string;
	items: NavItem[];
};

export const APP_NAV: NavSection[] = [
	{
		title: "Recommend a pipeline",
		caption: "Pick architecture, models, and params — export a JSON blueprint.",
		items: [
			{
				href: "/onboarding",
				label: "New recommendation",
				description: "Describe your use case and constraints",
				icon: IconClipboardList,
				matchPrefix: true,
			},
			{
				href: "/recommend",
				label: "Recommendation result",
				description: "Status, preliminary pick, and blueprint export",
				icon: IconReportAnalytics,
				matchPrefix: true,
			},
		],
	},
	{
		title: "Research lab",
		caption: "Try the RAG stack directly — not part of the recommendation flow.",
		items: [
			{
				href: "/chat",
				label: "Query corpus",
				description: "Chat with documents using the active pipeline",
				icon: IconMessageChatbot,
			},
			{
				href: "/compare",
				label: "Compare models",
				description: "Side-by-side LLM evaluation on your prompts",
				icon: IconArrowsLeftRight,
			},
		],
	},
];

export function isNavItemActive(pathname: string, item: NavItem): boolean {
	if (item.matchPrefix) {
		return pathname === item.href || pathname.startsWith(`${item.href}/`);
	}
	return pathname === item.href;
}

export function getPageMeta(pathname: string): {
	title: string;
	description: string;
} {
	for (const section of APP_NAV) {
		for (const item of section.items) {
			if (isNavItemActive(pathname, item)) {
				return { title: item.label, description: item.description };
			}
		}
	}
	return {
		title: "RecRAG",
		description: "RAG pipeline recommendation research tool",
	};
}
