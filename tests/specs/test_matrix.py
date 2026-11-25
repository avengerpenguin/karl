async def test_karl_replies_to_a_direct_message(alice, karl):
    sent_message = await alice.send_direct_message(
        karl,
        "What is two plus two?",
    )

    reply = await sent_message.wait_for_reply()

    assert reply.sender == karl
    assert "4" in reply.body or "four" in reply.body.lower()
