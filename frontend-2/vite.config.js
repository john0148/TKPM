import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Cho phép truy cập từ ngoài (vd: ngrok)
    port: 3000,
    allowedHosts: [
      '7fb0e2e4eb50.ngrok-free.app' // Thêm domain ngrok của bạn vào đây
    ],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
