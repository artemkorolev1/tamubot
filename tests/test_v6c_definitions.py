"""Verify v6c Definitions loads with the multi_asset and is queryable."""


def test_v6c_definitions_load():
    from tamubot.ingestion.pipeline_v6c.definitions import defs

    asset_keys = {key.to_user_string() for key in defs.resolve_asset_graph().get_all_asset_keys()}
    assert "v6c_bronze_markdown" in asset_keys
    assert "v6c_bronze_headers_sidecar" in asset_keys


def test_v6c_bronze_is_multi_asset():
    """Both v6c bronze outputs come from one compute (multi_asset)."""
    from tamubot.ingestion.pipeline_v6c.assets.bronze_odl import bronze_odl

    specs = list(bronze_odl.specs_by_key.keys())
    spec_names = {k.to_user_string() for k in specs}
    assert spec_names == {"v6c_bronze_markdown", "v6c_bronze_headers_sidecar"}
