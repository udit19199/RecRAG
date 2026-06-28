"use client";

import { usePathname, useRouter } from "next/navigation";
import { getPageTitle } from "@/components/layout/app-nav";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { Button } from "@/components/ui/button";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";

export function AppShell({ children }: { children: React.ReactNode }) {
	const router = useRouter();
	const pathname = usePathname();
	const title = getPageTitle(pathname);

	const signOut = async () => {
		await fetch("/api/auth/logout", { method: "POST" });
		router.replace("/sign-in");
		router.refresh();
	};

	return (
		<SidebarProvider defaultOpen>
			<AppSidebar />
			<SidebarInset className="min-h-svh bg-muted/20">
				<header className="sticky top-0 z-20 flex h-16 shrink-0 items-center justify-between gap-4 border-b bg-background/95 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/85">
					<div className="min-w-0">
						<h1 className="truncate text-sm font-semibold text-foreground">
							{title}
						</h1>
					</div>
					<Button
						type="button"
						variant="ghost"
						size="sm"
						className="shrink-0"
						onClick={signOut}
					>
						Sign out
					</Button>
				</header>

				<div className="flex min-h-0 flex-1 flex-col">{children}</div>
			</SidebarInset>
		</SidebarProvider>
	);
}
