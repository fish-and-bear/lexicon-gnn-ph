import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000 // Optional: specify dev server port
  },
  build: {
    outDir: 'dist' // Default output directory
  }
}) 