# Prosper Challenge

> **Implementation note:** This fork contains a fully-working solution to the challenge below.
> The bot collects patient name and date of birth, looks them up in Healthie via Playwright,
> then books the requested appointment — all in a live phone-style voice conversation.
>
> Key additions on top of the starter template:
> - Full scheduling conversation flow with LLM function calling (`find_patient`, `create_appointment`)
> - Playwright automation for Healthie patient search and appointment creation
> - Mid-call responsiveness: a side-channel responder answers questions while Playwright ops run
> - VAD tuning + STT commit strategy to prevent false turn-ends and merged transcripts
> - Session caching in `healthie.py` to avoid re-logging-in on every call
> - Unit tests in `tests/` for time normalisation and date validation logic

---

This is a template repository for an AI voice agent that is able to schedule appointments for a health clinic. To do that the agent connects in real-time to the clinic's CRM system, which in the healthcare industry is known as an Electronic Health Record (EHR). The foundations are already set:

- Pipecat is configured with sensible defaults and the bot already introduces itself when initialized
- Playwright is set up so that you can programmatically log into Healthie, the EHR we'll use for this challenge

However, for the agent to be fully functional you'll need to implement the following missig pieces:

- Expand the agent's configuration so that it asks for the patient's name and date of birth
- Once it finds the patient it should ask for the desired date and time of the appointment and create it
- Implement the find patient and create appointment functionalities using Playwright or otherwise

## Setup

To get started, fork this repository so that you can start commiting and pushing changes to your own copy.

### Prerequisites

#### Environment

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager installed

#### Healthie Account

You'll need a Healthie account for testing, you can create one [here](https://secure.gethealthie.com/users/sign_up/provider).

### Installation

1. Clone this repository

   ```bash
   git clone <repository-url>
   cd prosper-challenge
   ```

2. Copy the API keys we've shared with you, as well as your Healthie credentials:

   Create a `.env` file:

   ```bash
   cp env.example .env
   ```

   Then, add your API keys and credentials:

   ```ini
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   OPENAI_API_KEY=your_openai_api_key
   HEALTHIE_EMAIL=your_healthie_email
   HEALTHIE_PASSWORD=your_healthie_password
   ```

3. Set up a virtual environment and install dependencies

   ```bash
   uv sync
   ```

4. Install Playwright browsers

   ```bash
   uv run playwright install chromium
   ```

### Running the Bot

```bash
uv run bot.py
```

**Open http://localhost:7860 in your browser** and click `Connect` to start talking to your bot.

> 💡 First run note: The initial startup may take ~20 seconds as Pipecat downloads required models and imports.

### Running the Tests

```bash
uv run pytest tests/ -v
```

## Expectations & Deliverables

To make the agent functional we expect you to implement at least the following missing functionalities:

1. **Conversation Flow**: Modify the agent's behavior to ask for patient name and date of birth, then appointment date and time. [This guide](https://docs.pipecat.ai/guides/learn/function-calling) on function calling from Pipecat is probably a good start.

2. **Find Patient**: Implement `healthie.find_patient(name, date_of_birth)` in `healthie.py` to search for patients in Healthie.

3. **Create Appointment**: Implement `healthie.create_appointment(patient_id, date, time)` in `healthie.py` to create appointments in Healthie.

4. **Integration**: Connect the voice agent to these functions so it can actually schedule appointments during conversations.

We encourage you to use AI tools (Claude Code, Cursor, etc.) to help you with this challenge. We don't mind if you "vibe code" everything, that probably means you have good prompting skills. What we do care about is whether you understand the decisions and trade-offs behind your solution. That's why, apart from the code itself, we'd like you to write a high-level overview of your solution and the decisions you've made to get to it—do this in a `SOLUTION.md` file at the root of your fork. During the interview we'll dive deeper into it and discuss opportunities to improve it in the future.

If you'd like to go further, you can already document some of those potential improvements in your `SOLUTION.md`. Some areas that we'd love to hear your thoughts on are:
- Latency: balancing speed with user experience and accuracy
- Reliability: ensuring that the agent is always available to answer, regardless of external factors (e.g. AI provider unavailable)
- Evaluation: making it easy for us to check that the agent is behaving how it is supposed to


Once you are done, please share the link to your fork so that we can get familiar with it before our chat.
