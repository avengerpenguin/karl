async def test_karl_replies_to_a_direct_message(alice, karl_stub):
    sent_message = await alice.send_direct_message(
        karl_stub,
        "What is two plus two?",
    )

    reply = await sent_message.wait_for_reply()

    assert reply.sender == karl_stub
    assert "4" in reply.body or "four" in reply.body.lower()


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
