import path from "node:path";
import { fileURLToPath } from "node:url";
// @ts-ignore
import { loadEnvConfig } from "@next/env";
import type { NextConfig } from "next";

loadEnvConfig(path.join(path.dirname(fileURLToPath(import.meta.url)), ".."));

const nextConfig: NextConfig = {
  reactCompiler: true,
  output: "standalone",
  // Playwright/e2e often hits 127.0.0.1; without this, dev HMR breaks client hydration.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  env: {
    NEXT_PUBLIC_RETRIEVAL_API_URL: process.env.NEXT_PUBLIC_RETRIEVAL_API_URL,
    NEXT_PUBLIC_INGESTION_API_URL: process.env.NEXT_PUBLIC_INGESTION_API_URL,
  },
  async redirects() {
    return [
      { source: "/recommend", destination: "/recommendations", permanent: false },
    ];
  },
};

export default nextConfig;
