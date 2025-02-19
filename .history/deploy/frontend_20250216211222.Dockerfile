# Build stage
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy custom nginx config
COPY deploy/nginx/frontend.conf /etc/nginx/conf.d/default.conf

# Copy built assets
COPY --from=builder /app/build /usr/share/nginx/html

# Add health check
COPY deploy/scripts/health-check.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/health-check.sh

# Add security headers
RUN echo 'add_header X-Frame-Options "DENY";' >> /etc/nginx/conf.d/default.conf && \
    echo 'add_header X-Content-Type-Options "nosniff";' >> /etc/nginx/conf.d/default.conf && \
    echo 'add_header X-XSS-Protection "1; mode=block";' >> /etc/nginx/conf.d/default.conf && \
    echo 'add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";' >> /etc/nginx/conf.d/default.conf && \
    echo 'add_header Content-Security-Policy "default-src '\''self'\''; script-src '\''self'\'' '\''unsafe-inline'\''; style-src '\''self'\'' '\''unsafe-inline'\''; img-src '\''self'\'' data:; font-src '\''self'\''; connect-src '\''self'\''";' >> /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Use health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /usr/local/bin/health-check.sh

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 