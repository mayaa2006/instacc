// Minimal service worker: makes the app installable and opens instantly.
// API calls are never cached (always live data); the page shell falls back to cache when offline.
const SHELL = "instacc-shell-v1";
const FILES = ["/", "/icon.svg", "/icon-192.png", "/manifest.webmanifest"];
self.addEventListener("install", e => { e.waitUntil(caches.open(SHELL).then(c => c.addAll(FILES)).then(() => self.skipWaiting())); });
self.addEventListener("activate", e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== SHELL).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.startsWith("/api/") || url.pathname.startsWith("/auth/")) return;
  e.respondWith(fetch(e.request).then(r => {
    if (r.ok && FILES.includes(url.pathname)) { const copy = r.clone(); caches.open(SHELL).then(c => c.put(url.pathname, copy)); }
    return r;
  }).catch(() => caches.match(e.request.mode === "navigate" ? "/" : url.pathname)));
});
