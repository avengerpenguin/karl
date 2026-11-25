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


async def test_karl_with_tool_approved(alice, karl_with_hitl_tool):
    sent_message = await alice.send_direct_message(
        karl_with_hitl_tool,
        "What is the weather like in Berlin?",
    )

    notices = await sent_message.wait_for_notices()

    assert len(notices) == 2
    _thinking_notice, approval_notice = notices

    assert "Approval needed before continuing." in approval_notice.body

    approval_message = await alice.reply_to_message(approval_notice, "approve")
    reply = await approval_message.wait_for_reply()
    assert "20" in reply.body


async def test_karl_with_tool_rejected(alice, karl_with_hitl_tool):
    sent_message = await alice.send_direct_message(
        karl_with_hitl_tool,
        "What is the weather like in Madrid?",
    )

    notices = await sent_message.wait_for_notices()

    assert len(notices) == 2
    _thinking_notice, approval_notice = notices

    assert "Approval needed before continuing." in approval_notice.body

    rejection_message = await alice.reply_to_message(approval_notice, "deny")
    reply = await rejection_message.wait_for_reply()
    assert "20" not in reply.body


async def test_karl_with_tool_rejected_with_reason(alice, karl_with_hitl_tool):
    sent_message = await alice.send_direct_message(
        karl_with_hitl_tool,
        "What is the weather like in Rome?",
    )

    notices = await sent_message.wait_for_notices()

    assert len(notices) == 2
    _thinking_notice, approval_notice = notices

    assert "Approval needed before continuing." in approval_notice.body

    approval_message = await alice.reply_to_message(
        approval_notice, "deny: You must call the tool with country, e.g. Rome, Italy"
    )
    notices = await approval_message.wait_for_notices()

    assert len(notices) == 2
    resuming_notice, tool_notice = notices

    assert resuming_notice.body == "Resuming…"
    assert (
        tool_notice.body
        == "Calling tool `get_weather` with input `{'location': 'Rome, Italy'}`"
    )

    final_message = await approval_message.wait_for_reply()
    assert "20" in final_message.body
