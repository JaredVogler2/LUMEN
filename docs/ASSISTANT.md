# The optional assistant

LUMEN works completely without a language model, and ships with the assistant
**disabled**. Nothing in the measurement path uses it. This page is for an
administrator who is authorised to connect LUMEN to an approved,
OpenAI-compatible endpoint.

## What it may do

Explain results LUMEN has already computed: why a status is what it is, what
the failed checks mean, what to capture next, where a setting lives, and what a
pasted instruction text says.

## What it cannot do

It is given no tools and its reply is only displayed. It cannot compute or
change geometry, tolerances, gates, statuses or sessions, and it cannot decide
acceptance. Its text is shown under the label *"Assistant text. It explains
LUMEN's results; it is not a measurement and not an acceptance decision."*

A reply is discarded (the operator sees an error, not the text) unless:

* it is one JSON object with exactly the fields `answer`, `steps`,
  `citations`, `refusal`;
* every citation names a source that was in the request and quotes text that
  occurs in that source;
* every number with a unit in the reply already occurs in the request (so the
  model cannot introduce a coordinate, a tolerance or a measurement);
* it contains no statement of acceptance and no advice to ignore or override
  a check.

These checks are tested with deliberately bad replies
(`tests/integration/test_assistant.py`). They reduce, and do not remove, the
risk of a misleading explanation; the label above stays on every reply.

## What is sent to the endpoint

Only, per request:

* LUMEN's status, gate results (code, pass/fail, message, suggestion) and the
  uncertainty summary for the current view;
* the capture coaching items;
* the operator's typed question;
* instruction text the operator pasted, wrapped as quoted data with a
  provenance note. The model is told not to follow it.

Never sent: photos, the session id, the operator's name, tag sizes, device or
calibration data, operation records.

LUMEN stores neither the request nor the reply. The session log only records
that the assistant was used (kind, adapter, size of the request, number of
citations).

## Enabling it

Set these variables on the content item (Posit Connect: **Vars**):

| Variable | Value |
|---|---|
| `LUMEN_LLM_ENABLED` | `true` |
| `LUMEN_LLM_BASE_URL` | base address of the approved endpoint, e.g. `https://llm.example.org/v1`. HTTPS is required (plain HTTP is accepted for `localhost` only). The address cannot be set from the browser. |
| `LUMEN_LLM_MODEL` | model name understood by the endpoint |
| `LUMEN_LLM_API_KEY` | the key, as a secret variable. A different variable name can be given in `LUMEN_LLM_API_KEY_ENV`. The key is read on the server at request time, never sent to the browser, never logged. |
| `LUMEN_LLM_TIMEOUT_SECONDS` | optional, default `20` |

The endpoint must implement `POST {base}/chat/completions` with
`response_format: {"type": "json_object"}`.

## Retention and data handling at the endpoint

LUMEN controls what it sends, not what the endpoint keeps. Before enabling,
the administrator has to settle with the endpoint's owner, separately from
LUMEN: whether requests are logged or retained and for how long, whether they
may be used for training, where they are processed, and who may read them.
LUMEN has no setting for any of this.

## Testing

* `LUMEN_LLM_MOCK=1` together with `LUMEN_LLM_ENABLED=true` uses a built-in
  deterministic stand-in that restates the failed checks with citations. No
  network call is made. This is what the automated tests use.
* A real round trip is an opt-in test:
  `LUMEN_LLM_INTEGRATION=1 pytest -m llm_integration` with the variables above
  set. It has **not been run** in the development of LUMEN.
