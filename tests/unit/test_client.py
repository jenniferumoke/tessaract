
def test_model_name_is_split_into_prefix_and_name(client):

    assert client._normalize_model_name("oai/gpt-test") == ("oai", "gpt-test")