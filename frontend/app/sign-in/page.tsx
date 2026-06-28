"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";

type SignInResult = { ok: true } | { ok: false; error: string };

async function signIn(
	username: string,
	password: string,
): Promise<SignInResult> {
	try {
		const res = await fetch("/api/auth/login", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ username, password }),
		});
		if (!res.ok) {
			const data = (await res.json().catch(() => null)) as {
				error?: string;
			} | null;
			return { ok: false, error: data?.error ?? "Sign in failed" };
		}
		return { ok: true };
	} catch {
		return { ok: false, error: "Sign in failed" };
	}
}

function getRedirectPath(): string {
	const redirect = new URL(window.location.href).searchParams.get("redirect");
	return redirect ?? "/generate";
}

export default function SignInPage() {
	const router = useRouter();
	const [username, setUsername] = useState("");
	const [password, setPassword] = useState("");
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);

	const onSubmit = async (event: React.FormEvent) => {
		event.preventDefault();
		setLoading(true);
		setError(null);
		const result = await signIn(username, password);
		setLoading(false);
		if (!result.ok) {
			setError(result.error);
			return;
		}
		router.replace(getRedirectPath());
		router.refresh();
	};

	return (
		<div className="flex min-h-screen items-center justify-center p-4">
			<Card className="w-full max-w-sm">
				<CardHeader>
					<CardTitle>Sign in</CardTitle>
					<p className="text-sm text-muted-foreground">
						Internal research access for RecRAG.
					</p>
				</CardHeader>
				<CardContent>
					<form onSubmit={onSubmit} className="flex flex-col gap-4">
						<FieldGroup>
							<Field>
								<FieldLabel htmlFor="username">Username</FieldLabel>
								<Input
									id="username"
									autoComplete="username"
									value={username}
									onChange={(e) => setUsername(e.target.value)}
									required
								/>
							</Field>
							<Field>
								<FieldLabel htmlFor="password">Password</FieldLabel>
								<Input
									id="password"
									type="password"
									autoComplete="current-password"
									value={password}
									onChange={(e) => setPassword(e.target.value)}
									required
								/>
							</Field>
						</FieldGroup>
						{error ? <p className="text-sm text-destructive">{error}</p> : null}
						<Button type="submit" disabled={loading}>
							{loading ? "Signing in..." : "Sign in"}
						</Button>
					</form>
				</CardContent>
			</Card>
		</div>
	);
}
