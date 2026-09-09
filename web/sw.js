// Minimal service worker for the KrishiMitra AI PWA shell.
// Network-first for everything (this app is API-driven and shouldn't ever
// serve stale answers), falling back to a cached copy only when actually
// offline -- so the app shell (HTML/CSS/JS/icons) still opens without a
// connection, even though live chat obviously needs one.
const CACHE_NAME = 'krishimitra-shell-v1';
const SHELL_FILES = [
  '/ui/chatbot-ui-green.html',
  '/ui/dashboard-ui-green.html',
  '/ui/analytics-ui-green.html',
  '/ui/scheme-finder-ui-green.html',
  '/ui/documents-ui-green.html',
  '/ui/schemes-library-ui-green.html',
  '/ui/profile-ui-green.html',
  '/ui/manifest.json',
  '/ui/icons/icon-192.png',
  '/ui/icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return; // never cache POST /chat, /conversations, etc.

  event.respondWith(
    fetch(event.request)
      .then((res) => {
        // Only cache same-origin static shell files, never API responses.
        if (res.ok && SHELL_FILES.some((f) => event.request.url.endsWith(f))) {
          const copy = res.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
        }
        return res;
      })
      .catch(() => caches.match(event.request))
  );
});
