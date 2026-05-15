# RecRAG Frontend Dockerfile
#
# Builds a container for the Next.js frontend application

# Build stage
FROM node:22-alpine AS builder

# Set working directory
WORKDIR /app

# Build-time public API URLs for the browser bundle.
ARG NEXT_PUBLIC_RETRIEVAL_API_URL=http://localhost:8000
ARG NEXT_PUBLIC_INGESTION_API_URL=http://localhost:8001
ENV NEXT_PUBLIC_RETRIEVAL_API_URL=${NEXT_PUBLIC_RETRIEVAL_API_URL}
ENV NEXT_PUBLIC_INGESTION_API_URL=${NEXT_PUBLIC_INGESTION_API_URL}

# Copy package files
COPY frontend/package.json frontend/pnpm-lock.yaml ./

# Install dependencies
RUN corepack enable pnpm && pnpm install --frozen-lockfile

# Copy source code
COPY frontend/ ./

# Build the application
RUN pnpm build

# Production stage
FROM node:22-alpine AS runner

# Set working directory
WORKDIR /app

# Environment variables for production
ENV NODE_ENV=production

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy built artifacts
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Set environment variables
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Start the application
CMD ["node", "server.js"]
