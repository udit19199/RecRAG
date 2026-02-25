import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  output: "standalone",
  env: {
    NEXT_PUBLIC_RETRIEVAL_API_URL: process.env.NEXT_PUBLIC_RETRIEVAL_API_URL,
    NEXT_PUBLIC_INGESTION_API_URL: process.env.NEXT_PUBLIC_INGESTION_API_URL,
  },
};

export default nextConfig;
