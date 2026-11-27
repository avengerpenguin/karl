async def test_karl_shows_tool_calls_as_notice(alice, karl_with_tool):
    sent_message = await alice.send_direct_message(
        karl_with_tool,
        "What is the weather like in Paris?",
    )

    notices = await sent_message.wait_for_notices()
    reply = await sent_message.wait_for_reply()

    assert len(notices) == 2
    thinking_notice, tool_notice = notices

    assert thinking_notice.sender == karl_with_tool
    assert thinking_notice.body == "Thinking…"

    assert tool_notice.sender == karl_with_tool
    assert (
        tool_notice.body
        == "Calling tool `get_weather` with input `{'location': 'Paris'}`"
    )

    assert reply.sender == karl_with_tool
    assert "20" in reply.body


async def test_karl_reports_token_usage_in_final_notice(alice, karl_stub):
    sent_message = await alice.send_direct_message(
        karl_stub,
        "What is two plus two?",
    )

    reply = await sent_message.wait_for_reply()
    notices = await sent_message.wait_for_notices()

    assert reply.sender == karl_stub

    usage_notice = notices[-1]
    assert usage_notice.sender == karl_stub

    _model_part, separator, token_part = usage_notice.body.partition("tokens — ")
    assert separator, f"No token usage in notice: {usage_notice.body!r}"

    fields = dict(field.split(": ", maxsplit=1) for field in token_part.split(", "))

    assert fields.keys() == {"in", "out", "total"}
    assert all(value.isdigit() for value in fields.values())
    assert int(fields["total"]) == int(fields["in"]) + int(fields["out"])


async def test_karl_reports_model_in_final_notice(alice, karl_with_named_model):
    sent_message = await alice.send_direct_message(
        karl_with_named_model,
        "What is two plus two?",
    )

    await sent_message.wait_for_reply()
    notices = await sent_message.wait_for_notices()

    usage_notice = notices[-1]
    assert usage_notice.sender == karl_with_named_model
    assert usage_notice.body.startswith("model: stub-model-1 — ")
