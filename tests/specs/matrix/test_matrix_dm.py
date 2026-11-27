async def test_karl_replies_to_a_direct_message(alice, karl_stub):
    sent_message = await alice.send_direct_message(
        karl_stub,
        "What is two plus two?",
    )

    reply = await sent_message.wait_for_reply()

    assert reply.sender == karl_stub
    assert "4" in reply.body or "four" in reply.body.lower()
    assert sent_message.reactions == ["✅"]


async def test_karl_makes_failure_visible(alice, karl_broken):
    sent_message = await alice.send_direct_message(
        karl_broken,
        "What is two plus two?",
    )

    await sent_message.wait_for_reactions()
    assert sent_message.reactions == ["❌"]
    assert "went wrong" in sent_message.reply.body
