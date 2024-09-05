import tritonclient.grpc as grpcclient


def test_client():
    triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

    assert triton_client.is_server_live() is True
