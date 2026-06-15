import { SignJWT, jwtVerify } from "jose";

const COOKIE_NAME = "recrag_session";
const SESSION_MAX_AGE_SEC = 60 * 60 * 24 * 7; // 7 days

export { COOKIE_NAME, SESSION_MAX_AGE_SEC };

export function authEnabled(): boolean {
	return Boolean(process.env.RECRAG_AUTH_SECRET?.trim());
}

function secretKey(): Uint8Array {
	const secret = process.env.RECRAG_AUTH_SECRET;
	if (!secret?.trim()) {
		throw new Error("RECRAG_AUTH_SECRET is not configured");
	}
	return new TextEncoder().encode(secret);
}

export function verifyCredentials(username: string, password: string): boolean {
	const expectedUser = process.env.RECRAG_AUTH_USERNAME ?? "research";
	const expectedPass = process.env.RECRAG_AUTH_PASSWORD ?? "research";
	return username === expectedUser && password === expectedPass;
}

export async function createSessionToken(username: string): Promise<string> {
	return new SignJWT({ sub: username })
		.setProtectedHeader({ alg: "HS256" })
		.setIssuedAt()
		.setExpirationTime(`${SESSION_MAX_AGE_SEC}s`)
		.sign(secretKey());
}

export async function verifySessionToken(
	token: string,
): Promise<string | null> {
	if (!authEnabled()) {
		return "dev";
	}
	try {
		const { payload } = await jwtVerify(token, secretKey());
		const sub = payload.sub;
		return typeof sub === "string" ? sub : null;
	} catch {
		return null;
	}
}
