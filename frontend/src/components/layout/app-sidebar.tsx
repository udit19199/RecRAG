"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
	Sidebar,
	SidebarContent,
	SidebarFooter,
	SidebarGroup,
	SidebarGroupContent,
	SidebarGroupLabel,
	SidebarHeader,
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
	SidebarRail,
	SidebarTrigger,
} from "@/components/ui/sidebar";
import { APP_NAV, isNavItemActive } from "@/components/layout/app-nav";
import { cn } from "@/lib/utils";

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
							<div className="flex min-w-0 flex-1 flex-col group-data-[collapsible=icon]:hidden">
								<span className="truncate text-base font-semibold uppercase tracking-[0.12em]">
									RecRAG
								</span>
								<span className="truncate text-xs text-sidebar-foreground/60">
									Pipeline recommender
								</span>
							</div>
						</div>
					</SidebarMenuItem>
				</SidebarMenu>
			</SidebarHeader>

			<SidebarContent className="px-1.5 py-3">
				{APP_NAV.map((section) => (
					<SidebarGroup key={section.title}>
						<SidebarGroupLabel>{section.title}</SidebarGroupLabel>
						<p className="px-2 pb-2 text-xs leading-relaxed text-sidebar-foreground/55 group-data-[collapsible=icon]:hidden">
							{section.caption}
						</p>
						<SidebarGroupContent>
							<SidebarMenu>
								{section.items.map((item) => {
									const isActive = isNavItemActive(pathname, item);
									const Icon = item.icon;

									return (
										<SidebarMenuItem key={item.href}>
											<SidebarMenuButton
												asChild
												isActive={isActive}
												tooltip={`${item.label} — ${item.description}`}
												className={cn(
													"h-auto min-h-11 rounded-xl border border-transparent px-3 py-2.5 text-sm transition-[background-color,color,border-color] duration-150 group-data-[collapsible=icon]:size-10! group-data-[collapsible=icon]:min-h-0 group-data-[collapsible=icon]:rounded-2xl group-data-[collapsible=icon]:p-0! group-data-[collapsible=icon]:hover:bg-transparent",
													isActive
														? "border-sidebar-border bg-sidebar-accent/85 text-sidebar-foreground group-data-[collapsible=icon]:border-transparent group-data-[collapsible=icon]:bg-transparent"
														: "text-sidebar-foreground/72 hover:border-sidebar-border/80 hover:bg-sidebar-accent/80 hover:text-sidebar-foreground",
												)}
											>
												<Link
													href={item.href}
													className="flex w-full min-w-0 items-start justify-start gap-3 group-data-[collapsible=icon]:w-auto group-data-[collapsible=icon]:items-center group-data-[collapsible=icon]:justify-center"
												>
													<span
														className={cn(
															"mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg border border-transparent transition-[background-color,color,border-color] duration-150 group-data-[collapsible=icon]:mt-0",
															isActive
																? "border-sidebar-border bg-sidebar text-sidebar-foreground group-data-[collapsible=icon]:bg-sidebar-accent/95"
																: "text-sidebar-foreground/72 group-hover/menu-button:border-sidebar-border/80 group-hover/menu-button:bg-sidebar-accent/90 group-hover/menu-button:text-sidebar-foreground",
														)}
													>
														<Icon />
													</span>
													<span className="min-w-0 flex-1 group-data-[collapsible=icon]:hidden">
														<span className="block truncate text-sm font-medium leading-none">
															{item.label}
														</span>
														<span className="mt-1 block text-xs leading-snug text-sidebar-foreground/55">
															{item.description}
														</span>
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

			<SidebarFooter className="border-t border-sidebar-border/80 p-3 group-data-[collapsible=icon]:hidden">
				<p className="text-xs leading-relaxed text-sidebar-foreground/55">
					Start with <span className="text-sidebar-foreground/80">New recommendation</span>
					. Lab tools are for experimenting with ingestion and retrieval outside
					that flow.
				</p>
			</SidebarFooter>

			<SidebarRail />
		</Sidebar>
	);
}
