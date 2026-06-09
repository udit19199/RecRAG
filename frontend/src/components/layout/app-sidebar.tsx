"use client";

import { IconArrowsLeftRight, IconMessageChatbot } from "@tabler/icons-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
	Sidebar,
	SidebarContent,
	SidebarGroup,
	SidebarGroupContent,
	SidebarHeader,
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
	SidebarRail,
	SidebarTrigger,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";

const navigation = [
	{
		title: "Workspace",
		items: [
			{
				href: "/chat",
				label: "Chat",
				description: "Query indexed documents",
				icon: IconMessageChatbot,
			},
			{
				href: "/compare",
				label: "Compare",
				description: "Review candidate models",
				icon: IconArrowsLeftRight,
				badge: "New",
			},
		],
	},
];

export function AppSidebar() {
	const pathname = usePathname();

	return (
		<Sidebar collapsible="icon" className="border-r border-sidebar-border/80">
			<SidebarHeader className="gap-3 border-b border-sidebar-border/80 px-3 py-4">
				<SidebarMenu>
					<SidebarMenuItem>
						<div className="flex min-w-0 items-center gap-3">
							<SidebarTrigger
								className="flex size-8 shrink-0 items-center justify-center border border-sidebar-border bg-sidebar text-sidebar-foreground"
								aria-label="Toggle sidebar"
							/>

							<div className="flex min-w-0 flex-1 items-center">
								<span className="truncate text-base font-semibold uppercase tracking-[0.12em]">
									RecRAG
								</span>
							</div>
						</div>
					</SidebarMenuItem>
				</SidebarMenu>
			</SidebarHeader>

			<SidebarContent className="px-1.5 py-3">
				{navigation.map((group) => (
					<SidebarGroup key={group.title}>
						<SidebarGroupContent>
							<SidebarMenu>
								{group.items.map((item) => {
									const isActive = pathname === item.href;

									return (
										<SidebarMenuItem key={item.href}>
											<SidebarMenuButton
												asChild
												isActive={isActive}
												tooltip={item.label}
												className={cn(
													"h-11 rounded-xl border border-transparent px-3 py-2 text-sm transition-[background-color,color,border-color] duration-150 group-data-[collapsible=icon]:size-10! group-data-[collapsible=icon]:rounded-2xl group-data-[collapsible=icon]:p-0! group-data-[collapsible=icon]:hover:bg-transparent",
													isActive
														? "border-sidebar-border bg-sidebar-accent/85 text-sidebar-foreground group-data-[collapsible=icon]:border-transparent group-data-[collapsible=icon]:bg-transparent"
														: "text-sidebar-foreground/72 hover:border-sidebar-border/80 hover:bg-sidebar-accent/80 hover:text-sidebar-foreground",
												)}
											>
												<Link
													href={item.href}
													className="flex w-full min-w-0 items-center justify-start gap-3 group-data-[collapsible=icon]:w-auto group-data-[collapsible=icon]:justify-center"
												>
													<span
														className={cn(
															"flex size-8 shrink-0 items-center justify-center rounded-lg border border-transparent transition-[background-color,color,border-color,transform] duration-150",
															isActive
																? "border-sidebar-border bg-sidebar text-sidebar-foreground group-data-[collapsible=icon]:bg-sidebar-accent/95"
																: "text-sidebar-foreground/72 group-hover/menu-button:border-sidebar-border/80 group-hover/menu-button:bg-sidebar-accent/90 group-hover/menu-button:text-sidebar-foreground",
														)}
													>
														<item.icon />
													</span>
													<span className="truncate text-sm leading-none">
														{item.label}
													</span>
												</Link>
											</SidebarMenuButton>
										</SidebarMenuItem>
									);
								})}
							</SidebarMenu>
						</SidebarGroupContent>
					</SidebarGroup>
				))}
			</SidebarContent>

			<SidebarRail />
		</Sidebar>
	);
}
