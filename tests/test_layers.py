import torch

from timeseries_forge.models.layers import (
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
    causal_mask,
)


def test_glu_output_shape():
    glu = GatedLinearUnit(input_dim=10, output_dim=4)
    x = torch.randn(3, 7, 10)
    out = glu(x)
    assert out.shape == (3, 7, 4)


def test_grn_residual_path_when_dims_match():
    grn = GatedResidualNetwork(input_dim=8, hidden_dim=16, output_dim=8)
    x = torch.randn(2, 5, 8)
    out = grn(x)
    assert out.shape == (2, 5, 8)


def test_grn_projects_when_dims_differ():
    grn = GatedResidualNetwork(input_dim=8, hidden_dim=16, output_dim=12)
    x = torch.randn(2, 5, 8)
    out = grn(x)
    assert out.shape == (2, 5, 12)


def test_grn_with_context():
    grn = GatedResidualNetwork(input_dim=8, hidden_dim=16, output_dim=8, context_dim=4)
    x = torch.randn(2, 5, 8)
    ctx = torch.randn(2, 5, 4)
    out = grn(x, ctx)
    assert out.shape == (2, 5, 8)


def test_variable_selection_network_shapes_and_weights_sum_to_one():
    vsn = VariableSelectionNetwork(num_vars=6, hidden_dim=16)
    x = torch.randn(3, 10, 6)
    selected, weights = vsn(x)

    assert selected.shape == (3, 10, 16)
    assert weights.shape == (3, 10, 6)
    # softmax weights must sum to 1 across the variable dimension
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_variable_selection_network_rejects_wrong_num_vars():
    vsn = VariableSelectionNetwork(num_vars=6, hidden_dim=16)
    x = torch.randn(3, 10, 5)  # wrong number of variables
    try:
        vsn(x)
        assert False, "expected AssertionError for mismatched variable count"
    except AssertionError:
        pass


def test_attention_output_shape_and_weights_sum_to_one():
    attn = InterpretableMultiHeadAttention(d_model=16, n_heads=4)
    attn.eval()
    x = torch.randn(2, 9, 16)
    out, weights = attn(x)

    assert out.shape == (2, 9, 16)
    assert weights.shape == (2, 4, 9, 9)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_causal_mask_blocks_future_positions():
    mask = causal_mask(5, device=torch.device("cpu"))
    # row i should mask out all columns j > i
    for i in range(5):
        for j in range(5):
            if j > i:
                assert mask[i, j].item() is True
            else:
                assert mask[i, j].item() is False


def test_attention_with_causal_mask_does_not_nan():
    attn = InterpretableMultiHeadAttention(d_model=8, n_heads=2)
    x = torch.randn(2, 6, 8)
    mask = causal_mask(6, x.device)
    out, weights = attn(x, mask)
    assert not torch.isnan(out).any()
    assert not torch.isnan(weights).any()
