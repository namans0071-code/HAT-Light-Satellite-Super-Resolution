/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        linear: {
          bg: '#08090a',
          surface1: '#0f1011',
          surface2: '#141517',
          surface3: '#1c1d20',
          border: '#232529',
          borderLight: '#2e3138',
          textMuted: '#8a8f98',
          textSecondary: '#d0d6e0',
          textPrimary: '#f7f8f8',
          indigo: '#5e6ad2',
          indigoHover: '#6e7be0',
          emerald: '#10b981',
          amber: '#f59e0b',
          violet: '#8b5cf6',
          cyan: '#06b6d4',
          rose: '#f43f5e'
        }
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        'linear-glow': '0 0 20px -5px rgba(94, 106, 210, 0.3)',
        'linear-card': '0 4px 24px -1px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.06)',
        'linear-modal': '0 20px 50px -10px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.08)',
      }
    },
  },
  plugins: [],
};
