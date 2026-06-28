import path from "node:path";
import { fileURLToPath } from "node:url";
// @ts-ignore
import { loadEnvConfig } from "@next/env";
import type { NextConfig } from "next";

loadEnvConfig(path.join(path.dirname(fileURLToPath(import.meta.url)), ".."));

const nextConfig: NextConfig = {
  reactCompiler: true,
  output: "standalone",
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
