"use client";

import { useState } from "react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

interface WorkspaceSetupProps {
	onComplete: () => void;
	onPrev: () => void;
}

export function WorkspaceSetup({ onComplete, onPrev }: WorkspaceSetupProps) {
	const [workspaceName, setWorkspaceName] = useState("");
	const [teammates, setTeammates] = useState("");
	const [message, setMessage] = useState("Come join my RecRAG workspace!");

	const handleWorkspaceNameKeyDown = (
		event: React.KeyboardEvent<HTMLInputElement>,
	) => {
		if (
			(event.metaKey || event.ctrlKey) &&
			event.key === "Enter" &&
			workspaceName.trim()
		) {
			onComplete();
		}
	};

	return (
		<div className="flex animate-in flex-col duration-300 fade-in">
			<div className="-mx-8 -mt-8 mb-8 flex items-center justify-between border-b bg-muted/30 px-8 py-6 md:-mx-12 md:-mt-12 md:px-12">
				<div className="flex items-center gap-3">
					<Avatar size="sm">
						<AvatarFallback className="rounded-sm bg-foreground font-bold text-background text-xs">
							{workspaceName ? workspaceName.charAt(0).toUpperCase() : "W"}
						</AvatarFallback>
					</Avatar>
					<span className="font-mono text-sm font-medium text-foreground">
						workspace_init.sh
					</span>
				</div>
			</div>

			<div className="mb-8 text-left">
				<h1 className="mb-1 text-2xl font-semibold tracking-tight text-foreground">
					Configure Workspace
				</h1>
				<p className="text-sm text-muted-foreground">
					Initialize your retrieval pipelines and configure teammates.
				</p>
			</div>

			<FieldGroup>
				<Field>
					<FieldLabel htmlFor="workspaceName">Name your workspace</FieldLabel>
					<Input
						id="workspaceName"
						type="text"
						placeholder="e.g. Acme Corp Knowledge Base"
						value={workspaceName}
						onChange={(e) => setWorkspaceName(e.target.value)}
						onKeyDown={handleWorkspaceNameKeyDown}
						className="font-mono"
					/>
				</Field>

				<Field>
					<FieldLabel htmlFor="teammates">
						Add your teammates (Optional)
					</FieldLabel>
					<Input
						id="teammates"
						type="text"
						placeholder="Type names or emails..."
						value={teammates}
						onChange={(e) => setTeammates(e.target.value)}
						className="font-mono"
					/>
				</Field>

				{teammates.length > 0 ? (
					<Field className="animate-in fade-in slide-in-from-top-2">
						<FieldLabel htmlFor="message">Custom message</FieldLabel>
						<Textarea
							id="message"
							rows={3}
							value={message}
							onChange={(e) => setMessage(e.target.value)}
						/>
					</Field>
				) : null}
			</FieldGroup>

			<div className="mt-10 flex items-center justify-between">
				<Button type="button" variant="ghost" onClick={onPrev}>
					Back
				</Button>
				<Button
					type="button"
					onClick={onComplete}
					disabled={!workspaceName.trim()}
				>
					{teammates ? "Invite & Finish" : "Finish Setup"}
				</Button>
			</div>
		</div>
	);
}
