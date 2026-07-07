import {
	IconArrowsLeftRight,
	IconHistory,
	IconSparkles,
	IconTopologyStar3,
} from "@tabler/icons-react";
import type { ComponentType } from "react";

export type NavItem = {
	href: string;
	label: string;
	icon: ComponentType<{ className?: string }>;
	matchPrefix?: boolean;
};

export const APP_NAV: NavItem[] = [
	{
		href: "/compare",
		label: "Compare",
		icon: IconArrowsLeftRight,
	},
	{
		href: "/graphrag",
		label: "GraphRAG",
		icon: IconTopologyStar3,
	},
	{
		href: "/generate",
		label: "Generate",
		icon: IconSparkles,
	},
	{
		href: "/recommendations",
		label: "History",
		icon: IconHistory,
		matchPrefix: true,
	},
];

export function isNavItemActive(pathname: string, item: NavItem): boolean {
	if (item.matchPrefix) {
		return pathname === item.href || pathname.startsWith(`${item.href}/`);
	}
	return pathname === item.href;
}

export function getPageTitle(pathname: string): string {
	for (const item of APP_NAV) {
		if (isNavItemActive(pathname, item)) {
			return item.label;
		}
	}
	return "RecRAG";
}
