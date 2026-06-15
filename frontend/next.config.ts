import path from "node:path";
import { fileURLToPath } from "node:url";
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
};

export default nextConfig;
