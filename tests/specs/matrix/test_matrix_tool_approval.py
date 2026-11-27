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


async def test_karl_with_multiple_tool_approvals(alice, karl_with_two_hitl_tools):
    sent_message = await alice.send_direct_message(
        karl_with_two_hitl_tools,
        "What is the current time and current weather in Lisbon?",
    )

    notices = await sent_message.wait_for_notices()

    assert len(notices) == 2
    _thinking_notice, approval_notice = notices

    assert "This approval contains **2 tool calls**" in approval_notice.body
    assert "`get_weather`" in approval_notice.body
    assert "`get_time`" in approval_notice.body

    approval_message = await alice.reply_to_message(approval_notice, "approve")
    reply = await approval_message.wait_for_reply()
    assert "20" in reply.body
