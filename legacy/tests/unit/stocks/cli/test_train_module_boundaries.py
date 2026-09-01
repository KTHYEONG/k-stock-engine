def test_train_cli_request_builder_is_explicitly_wired() -> None:
    from legacy.stocks.cli import train_request

    parsed = train_request.build_parser().parse_args(["--artifact-id", "boundary-test"])
    assert callable(train_request._build_training_request)
    assert parsed.artifact_id == "boundary-test"
