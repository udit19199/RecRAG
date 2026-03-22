"use client";

import { AppSidebar } from "@/components/layout/app-sidebar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";

export function AppShell({ children }: { children: React.ReactNode }) {
	return (
		<SidebarProvider defaultOpen>
			<AppSidebar />
			<SidebarInset className="min-h-svh bg-muted/20">
				<header className="sticky top-0 z-20 flex h-16 shrink-0 items-center justify-center border-b bg-background/95 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/85">
					<h1 className="text-sm font-semibold uppercase tracking-[0.14em] text-foreground">
						RecRAG
					</h1>
				</header>

				<div className="flex min-h-0 flex-1 flex-col">{children}</div>
			</SidebarInset>
		</SidebarProvider>
	);
}
