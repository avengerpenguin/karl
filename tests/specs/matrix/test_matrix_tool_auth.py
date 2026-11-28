async def test_karl_requests_oauth_authorization_in_the_same_thread(
    alice, karl_needing_authorization
):
    prompt = await alice.send_direct_message(
        karl_needing_authorization,
        "Show my calendar today.",
    )

    notices = await prompt.wait_for_notices()

    authorization_notice = notices[-1]
    assert authorization_notice.sender == karl_needing_authorization
    assert "Sign in to continue" in authorization_notice.body
    assert karl_needing_authorization.authorization_url in authorization_notice.body

    await alice.visit(karl_needing_authorization.authorization_url)

    reply = await prompt.wait_for_reply()
    assert reply.sender == karl_needing_authorization
