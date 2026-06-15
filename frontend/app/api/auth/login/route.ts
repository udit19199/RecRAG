import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import {
	COOKIE_NAME,
	SESSION_MAX_AGE_SEC,
	authEnabled,
	createSessionToken,
	verifyCredentials,
	verifySessionToken,
} from "@/lib/auth/session";

export async function POST(request: Request) {
	if (!authEnabled()) {
		return NextResponse.json(
			{ error: "Auth is disabled in this environment" },
			{ status: 503 },
		);
	}

	const body = (await request.json()) as {
		username?: string;
		password?: string;
	};
	const username = body.username?.trim() ?? "";
	const password = body.password ?? "";

	if (!verifyCredentials(username, password)) {
		return NextResponse.json(
			{ error: "Invalid username or password" },
			{ status: 401 },
		);
	}

	const token = await createSessionToken(username);
	const response = NextResponse.json({ username });
	response.cookies.set(COOKIE_NAME, token, {
		httpOnly: true,
		sameSite: "lax",
		secure: process.env.NODE_ENV === "production",
		path: "/",
		maxAge: SESSION_MAX_AGE_SEC,
	});
	return response;
}

export async function GET() {
	if (!authEnabled()) {
		return NextResponse.json({ username: "dev", auth: false });
	}
	const token = (await cookies()).get(COOKIE_NAME)?.value;
	if (!token) {
		return NextResponse.json({ username: null, auth: true }, { status: 401 });
	}
	const username = await verifySessionToken(token);
	if (!username) {
		return NextResponse.json({ username: null, auth: true }, { status: 401 });
	}
	return NextResponse.json({ username, auth: true });
}
