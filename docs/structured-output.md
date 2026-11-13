# Structured output

Use llm's native schema options with Devin:

```bash
llm -m devin --schema 'title, risk int' "Assess this change"
llm -m devin --schema-multi 'title, risk int' "Assess each change"
llm -m devin --schema schema.json "Assess this change"
```

`--schema-multi` wraps the requested fields in an object with an `items` array.
A JSON schema file should contain a self-contained JSON Schema Draft 7 schema
(the API limit is 64 KB).

For a new session, the plugin sends the schema as `structured_output_schema`
in the create-session request, without changing the prompt text. It leaves
`structured_output_required` at the API default (true): Devin must provide
final structured output before the turn ends. The plugin reads the validated
`structured_output` from the final GET session response.

In schema mode stdout contains only the JSON value, not Devin chat messages.
Session URLs, continuation notices, and debug-log paths go to stderr. Without
`--schema`, the existing text behavior is unchanged. Logged `response_json`
contains `session_id`, `end_cursor`, `structured_output_schema`, and
`structured_output` for successful schema turns.

## Continuing a session

Continue with `llm -c --schema 'title, risk int' "Follow up"` using exactly the
same schema (dictionary equality) as the session was created with. The schema
must be recorded in local conversation history, including history loaded from
the logs database. The Devin API has no documented endpoint to change a
session's structured output schema after creation. A different or unrecorded
schema is rejected before any HTTP request. In particular, `-o session` plus
`--schema` cannot target a session without matching local schema history.
Start a new conversation with `--schema` instead, or omit `--schema` to continue
in plain text.

For schema continuations the plugin reads the existing output before sending
the message. If the final output is missing or unchanged, it raises an error
rather than emitting stale JSON. Even an intentionally identical result is
indistinguishable from stale output; inspect it with `llm devin status <url>`.
New sessions that end without structured output also raise an error including
the status and session URL. An error-status session always raises, even if it
has structured output. Empty objects, arrays, false, and zero are emitted;
API `null` is indistinguishable from output not yet produced and is an error.

See the [Devin API reference](https://docs.devin.ai/api-reference/overview)
and [official v3 OpenAPI schema](https://docs.devin.ai/v3-openapi.yaml).
