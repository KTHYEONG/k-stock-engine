def test_contract_facade_preserves_training_request_identity() -> None:
    from src.stocks.ml import contracts
    from src.stocks.ml import training_contracts

    assert issubclass(training_contracts.NetAlphaTrainingRequest, contracts.NetAlphaTrainingRequest)
    assert contracts.NetAlphaTrainingRequest.__dataclass_params__.frozen is True
