# Solution Overview

## What I Built

I ended up with a voice agent that lets someone call in (via the browser), say their name and date of birth, and get an appointment booked in Healthie without touching the keyboard. It handles back-and-forth conversation, recovers when STT messes up a name, and stays responsive during the 5–15 second Healthie operations instead of going silent.

---

## Architecture

```
User (browser mic/speaker)
        ↕  WebRTC audio
    bot.py  — Pipecat pipeline
        ├── ElevenLabs Realtime STT   (speech → text, streaming)
        ├── Smart Turn Analyzer V3    (end-of-turn detection, local ONNX model)
        ├── Silero VAD                (voice activity detection)
        ├── OpenAI LLM                (conversation + function calling; Pipecat default, e.g. gpt-4.1)
        │       ↕  tool calls (non-cancellable)
        │   healthie.py
        │       ├── login_to_healthie()    singleton session + cookie persistence
        │       ├── find_patient()         search + DOB verification
        │       └── create_appointment()   modal form automation via Playwright
        └── ElevenLabs TTS            (text → speech, streaming)
```

I didn’t set a model for the main pipeline — it uses whatever Pipecat’s `OpenAILLMService` defaults to (in the version I have that’s `gpt-4.1`). The mid-wait responder is explicitly `gpt-4o-mini` so hold replies are cheap and fast. If the template or your Pipecat defaults to something older, upgrading to newer models when available is an easy win for cost and latency.

---

## Key Decisions

### 1. Playwright instead of the Healthie API

Healthie has a GraphQL API but it needs enterprise credentials. On the free sandbox I only had email/password for the web app, so I stuck with Playwright like the starter suggested.

The downside is latency: every Healthie action is 3–15 seconds instead of a couple hundred ms. A lot of what I did was just coping with that — session reuse, filler phrases, answering questions while the lookup runs, etc.

### 2. One browser session and cookie persistence

I noticed that spinning up a new browser for every `find_patient` cost ~5s before anything useful happened. So I kept a single `Browser` / `Context` / `Page` in module-level variables and reuse them. After a successful login I save cookies to `.healthie_session.json`; on the next run I load them so Healthie thinks it’s the same device and often skips login.

That took cold login from ~20s down to ~0s when the session is still valid, and ~1–2s when we’re loading cookies on a fresh process.

### 3. Pre-warming at startup

Even with cookies, the first real call after a restart would still hit a cold browser. So I call `login_to_healthie()` in `bot()` before any user connects. By the time someone is on the line, the session is already warm and the first lookup doesn’t make them wait 20 seconds.

### 4. Waiting on the DOM instead of fixed sleeps

My first version had a bunch of `page.wait_for_timeout()` calls “just in case” the React app wasn’t ready — something like 18 seconds of sleep on every run, even when we were reusing the session. I replaced all of that with `wait_for_selector()` on things that actually mean “ready”:

- `[data-testid="main-nav"]` — only there when we’re logged in
- `input[placeholder="Search Clients..."]` — search bar is mounted
- `a[href*="/users/"]` — search results are back
- `[role="dialog"]` hidden — modal closed after submit

Everything finishes as soon as the page is actually ready instead of waiting a worst-case delay every time.

### 5. Staying conversational during EHR operations

While Playwright is running, the main pipeline LLM is blocked waiting for the tool result — so the bot would go silent for 5–15 seconds. I tried a few things (fake tool result + background work, timers with filler phrases) but ran into OpenAI’s strict ordering: you can’t send a second `tool` message for the same call later. So the call has to stay blocking.

What I did instead: a `TranscriptInterceptor` between STT and the user aggregator. When a tool is running, it intercepts committed transcripts, puts them in a queue, and doesn’t push them to the main pipeline (so we don’t get duplicate turns). A separate `_respond_while_waiting` coroutine reads that queue and does a direct `gpt-4o-mini` call to answer things like “Are you still there?” or “How long does this take?” — the reply goes straight to TTS. When the tool finishes, I inject those user/assistant pairs into the main context so the post-tool LLM doesn’t answer the same questions again.

So the user gets a real answer in ~500ms instead of dead air. I used gpt-4o-mini for that side channel — cheaper and faster. The main pipeline uses Pipecat’s default (no model passed in code); in my setup that’s gpt-4.1. If your Pipecat or the original template defaults to something older, it’s worth calling that out and switching to newer models when they’re available for better cost and latency.

### 6. Not cancelling tool calls on interruption

By default, if the user talks while a tool is running, Pipecat cancels the tool. For `create_appointment` that’s risky: we could leave the Healthie form half-filled. I registered both tools with `cancel_on_interruption=False` so the Playwright run always completes; the LLM can handle whatever they said on the next turn.

### 7. DOB when it’s there

Healthie search is by name only. I go to the profile after finding a match and compare DOB if the profile has it; if there’s no DOB on file (common in sandbox), I accept the name match. So we’re not inventing a requirement — we use DOB when available to double-check.

### 8. STT name mess-ups

I kept hitting cases where the STT split a name (e.g. “Harikumar” → “Hari Kumar”). Two things helped: (1) `find_patient` retries with just the first name if the full name returns nothing, and (2) the system prompt tells the LLM to ask for spelling and retry if the first lookup fails.

### 9. VAD and STT commit tuning

Out of the box, the VAD was firing on fan noise and keystrokes, so I got phantom user turns and weird duplicates. I tuned Silero (`confidence=0.8`, `start_secs=0.3`, `stop_secs=0.6`, `min_volume=0.75`) so we need a bit more real speech before we start a turn, but still catch short “Yes” / “Yeah” quickly. For ElevenLabs STT I use `CommitStrategy.MANUAL` with `vad_silence_threshold_secs=0.8` so we commit when our VAD says the user stopped, and we don’t merge a short confirmation with whatever noise comes after.

---

## What I’d Improve Next

**Latency:** The biggest win would be using Healthie’s API instead of Playwright — would drop each op to ~200ms. Session keep-alive (e.g. hit the home page every 20 min) would avoid cold starts after idle. I could also stream multiple hold phrases on a timer, or cache patient ID by name for repeat callers. **Model choice:** We’re not hardcoding the main LLM — it uses Pipecat’s default. The mid-wait responder is gpt-4o-mini. As newer models ship (e.g. newer GPT or other providers), we could explicitly switch to whatever is cheaper or faster for each role and mention that as an easy upgrade path.

**Reliability:** Retry with backoff around Playwright calls; if things keep failing, have the bot offer to take a message and have someone call back. A fallback LLM (e.g. Anthropic) would help when the primary provider is down.

**Evaluation:** Unit tests cover time normalisation and date parsing. The next step I’d add is a harness that injects text into the pipeline (no STT) and drives full flows in CI. Plus structured logging of every find_patient/create_appointment (inputs, result, latency) and dashboards on Pipecat’s TTFB/stage metrics so we can see when EHR or STT goes sideways.
