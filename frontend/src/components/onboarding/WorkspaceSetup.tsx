"use client";

import { useState } from "react";

interface WorkspaceSetupProps {
	onComplete: () => void;
	onPrev: () => void;
}

export function WorkspaceSetup({ onComplete, onPrev }: WorkspaceSetupProps) {
	const [workspaceName, setWorkspaceName] = useState("");
	const [teammates, setTeammates] = useState("");
	const [message, setMessage] = useState("Come join my RecRAG workspace!");

	const handleKeyDown = (e: React.KeyboardEvent) => {
		if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && workspaceName.trim()) {
			onComplete();
		}
	};

	return (
		<form
			className="flex flex-col animate-in fade-in duration-300"
			onKeyDown={handleKeyDown}
			onSubmit={(e) => e.preventDefault()}
		>
			<div className="w-full border-b border-border -mt-8 md:-mt-12 -mx-8 md:-mx-12 px-8 md:px-12 py-6 mb-8 bg-muted/30 flex items-center justify-between">
				<div className="flex items-center gap-3">
					<div className="w-8 h-8 bg-foreground rounded-sm flex items-center justify-center text-background font-bold text-xs">
						{workspaceName ? workspaceName.charAt(0).toUpperCase() : "W"}
					</div>
					<span className="font-mono text-sm font-medium text-foreground">
						workspace_init.sh
					</span>
				</div>
			</div>

			<div className="text-left mb-8">
				<h1 className="text-2xl font-semibold text-foreground mb-1 tracking-tight">
					Configure Workspace
				</h1>
				<p className="text-sm text-muted-foreground">
					Initialize your retrieval pipelines and configure teammates.
				</p>
			</div>

			<div className="flex flex-col gap-6">
				<div className="flex flex-col gap-2 text-left">
					<label
						htmlFor="workspaceName"
						className="text-sm font-medium text-foreground"
					>
						Name your workspace
					</label>
					<input
						id="workspaceName"
						type="text"
						placeholder="e.g. Acme Corp Knowledge Base"
						value={workspaceName}
						onChange={(e) => setWorkspaceName(e.target.value)}
						className="w-full p-2.5 bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-foreground focus:border-foreground transition-colors text-foreground placeholder:text-muted-foreground font-mono text-sm"
					/>
				</div>

				<div className="flex flex-col gap-2 text-left">
					<label
						htmlFor="teammates"
						className="text-sm font-medium text-foreground"
					>
						Add your teammates (Optional)
					</label>
					<input
						id="teammates"
						type="text"
						placeholder="Type names or emails..."
						value={teammates}
						onChange={(e) => setTeammates(e.target.value)}
						className="w-full p-2.5 bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-foreground focus:border-foreground transition-colors text-foreground placeholder:text-muted-foreground font-mono text-sm"
					/>
				</div>

				{teammates.length > 0 && (
					<div className="flex flex-col gap-2 text-left animate-in fade-in slide-in-from-top-2">
						<label
							htmlFor="message"
							className="text-sm font-medium text-foreground"
						>
							Custom message
						</label>
						<textarea
							id="message"
							rows={3}
							value={message}
							onChange={(e) => setMessage(e.target.value)}
							className="w-full p-2.5 bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-foreground focus:border-foreground transition-colors resize-none text-foreground text-sm"
						/>
					</div>
				)}
			</div>

			<div className="mt-10 flex items-center justify-between">
				<button
					type="button"
					onClick={onPrev}
					className="text-muted-foreground hover:text-foreground text-sm font-medium transition-colors"
				>
					Back
				</button>
				<button
					type="button"
					onClick={onComplete}
					className="bg-foreground hover:bg-foreground/90 text-background px-6 py-2 rounded-md font-medium transition-colors disabled:opacity-50"
					disabled={!workspaceName.trim()}
				>
					{teammates ? "Invite & Finish" : "Finish Setup"}
				</button>
			</div>
		</form>
	);
}
