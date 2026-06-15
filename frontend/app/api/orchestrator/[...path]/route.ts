import { cookies } from "next/headers";
import { type NextRequest, NextResponse } from "next/server";
import { authEnabled, verifySessionToken } from "@/lib/auth/session";

const ORCHESTRATOR_URL =
	process.env.ORCHESTRATOR_API_URL ??
	process.env.NEXT_PUBLIC_ORCHESTRATOR_API_URL ??
	"http://127.0.0.1:8002";

async function proxy(request: NextRequest, path: string[]) {
	const target = new URL(path.join("/"), `${ORCHESTRATOR_URL}/`);
	request.nextUrl.searchParams.forEach((value, key) => {
		target.searchParams.set(key, value);
	});

	const headers = new Headers();
	const contentType = request.headers.get("content-type");
	if (contentType) {
		headers.set("content-type", contentType);
	}

	if (authEnabled()) {
		const token = (await cookies()).get("recrag_session")?.value;
		if (!token) {
			return NextResponse.json({ detail: "Sign in required" }, { status: 401 });
		}
		const username = await verifySessionToken(token);
		if (!username) {
			return NextResponse.json(
				{ detail: "Invalid or expired session" },
				{ status: 401 },
			);
		}
		headers.set("authorization", `Bearer ${token}`);
	}

	const init: RequestInit = {
		method: request.method,
		headers,
	};
	if (request.method !== "GET" && request.method !== "HEAD") {
		init.body = await request.text();
	}

	const upstream = await fetch(target, init);
	const body = await upstream.text();
	return new NextResponse(body, {
		status: upstream.status,
		headers: {
			"content-type":
				upstream.headers.get("content-type") ?? "application/json",
		},
	});
}

type RouteContext = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, context: RouteContext) {
	const { path } = await context.params;
	return proxy(request, path);
}

export async function POST(request: NextRequest, context: RouteContext) {
	const { path } = await context.params;
	return proxy(request, path);
}

export async function PUT(request: NextRequest, context: RouteContext) {
	const { path } = await context.params;
	return proxy(request, path);
}

export async function PATCH(request: NextRequest, context: RouteContext) {
	const { path } = await context.params;
	return proxy(request, path);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
	const { path } = await context.params;
	return proxy(request, path);
}
