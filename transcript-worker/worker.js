const ALLOWED_ORIGINS = [
  "https://youtube-chatbot-api.onrender.com",
  "http://localhost:8000",
];

export default {
  async fetch(request) {
    const origin = request.headers.get("Origin") || "";
    const corsHeaders = {
      "Access-Control-Allow-Origin": ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0],
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    if (request.method !== "POST") {
      return Response.json({ error: "POST only" }, { status: 405, headers: corsHeaders });
    }

    try {
      const { videoId } = await request.json();
      if (!videoId) {
        return Response.json({ error: "videoId required" }, { status: 400, headers: corsHeaders });
      }

      const segments = await fetchTranscript(videoId);
      if (!segments) {
        return Response.json({ error: "No transcript found" }, { status: 404, headers: corsHeaders });
      }

      return Response.json({ videoId, segments }, { headers: corsHeaders });
    } catch (err) {
      return Response.json({ error: err.message }, { status: 500, headers: corsHeaders });
    }
  },
};

async function fetchTranscript(videoId) {
  // Step 1: Get caption track URLs via Innertube API
  const playerResp = await fetch("https://www.youtube.com/youtubei/v1/player", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      context: {
        client: { hl: "en", gl: "US", clientName: "WEB", clientVersion: "2.20250320.01.00" },
      },
      videoId,
    }),
  });

  const playerData = await playerResp.json();
  const tracks =
    playerData?.captions?.playerCaptionsTracklistRenderer?.captionTracks || [];

  if (!tracks.length) {
    // Fallback: scrape the watch page for caption tracks
    return await fetchViaPageScrape(videoId);
  }

  const track = tracks.find((t) => t.languageCode === "en") || tracks[0];
  return await fetchCaptionTrack(track.baseUrl);
}

async function fetchViaPageScrape(videoId) {
  const resp = await fetch(`https://www.youtube.com/watch?v=${videoId}`, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
      Cookie: "CONSENT=PENDING+987;",
    },
  });

  const html = await resp.text();
  const match = html.match(/"captionTracks":\s*(\[.*?\])/);
  if (!match) return null;

  const tracks = JSON.parse(match[1]);
  if (!tracks.length) return null;

  const track = tracks.find((t) => t.languageCode === "en") || tracks[0];
  return await fetchCaptionTrack(track.baseUrl);
}

async function fetchCaptionTrack(baseUrl) {
  const sep = baseUrl.includes("?") ? "&" : "?";
  const resp = await fetch(baseUrl + sep + "fmt=json3");
  const data = await resp.json();

  if (!data.events) return null;

  return data.events
    .filter((e) => e.segs)
    .map((e) => ({
      text: e.segs.map((s) => s.utf8 || "").join(""),
      start: (e.tStartMs || 0) / 1000,
      duration: (e.dDurationMs || 0) / 1000,
    }))
    .filter((s) => s.text.trim());
}
