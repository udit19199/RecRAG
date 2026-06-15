import { type NextRequest, NextResponse } from "next/server";
import { COOKIE_NAME, authEnabled, verifySessionToken } from "@/lib/auth/session";

const PUBLIC_PATHS = ["/", "/sign-in"];

export async function middleware(request: NextRequest) {
	if (!authEnabled()) {
		return NextResponse.next();
	}

	const { pathname } = request.nextUrl;
	if (
		PUBLIC_PATHS.includes(pathname) ||
		pathname.startsWith("/api/auth/") ||
		pathname.startsWith("/_next/")
	) {
		return NextResponse.next();
	}

	const token = request.cookies.get(COOKIE_NAME)?.value;
	const username = token ? await verifySessionToken(token) : null;
	if (username) {
		return NextResponse.next();
	}

	if (pathname.startsWith("/api/")) {
		return NextResponse.json({ detail: "Sign in required" }, { status: 401 });
	}

	const signIn = new URL("/sign-in", request.url);
	signIn.searchParams.set("redirect", pathname);
	return NextResponse.redirect(signIn);
}

export const config = {
	matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
