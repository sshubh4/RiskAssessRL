/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'tv-bg':      '#131722',
        'tv-panel':   '#1e2329',
        'tv-card':    '#2a2e39',
        'tv-hover':   '#363c4e',
        'tv-text':    '#d1d4dc',
        'tv-muted':   '#787b86',
        'tv-dim':     '#4c525e',
        'tv-green':   '#26a69a',
        'tv-red':     '#ef5350',
        'tv-blue':    '#2962ff',
        'tv-border':  '#2a2e39',
      },
    },
  },
  plugins: [],
}
