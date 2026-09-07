"""Model selection across OpenAI, Anthropic and DeepSeek.

The registry is pure data, so every routing rule can be checked without constructing a
client or holding an API key for any of the three.
"""

import pytest

import providers
from providers import UnknownProviderError


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o", "openai"),
        ("gpt-4o-mini", "openai"),
        ("o3-mini", "openai"),
        ("chatgpt-4o-latest", "openai"),
        ("claude-sonnet-5", "anthropic"),
        ("claude-opus-5", "anthropic"),
        ("claude-haiku-4-5-20251001", "anthropic"),
        ("deepseek-chat", "deepseek"),
        ("deepseek-reasoner", "deepseek"),
        # Case is not meaningful in a model id for routing purposes.
        ("Claude-Sonnet-5", "anthropic"),
        ("  deepseek-chat  ", "deepseek"),
    ],
)
def test_provider_is_inferred_from_the_model_id(model: str, expected: str) -> None:
    assert providers.provider_for_model(model) == expected


@pytest.mark.parametrize("model", ["", "   ", "some-unreleased-model"])
def test_unrecognised_models_fall_back_to_the_default_provider(model: str) -> None:
    # A guess is better than a crash here; LLM_PROVIDER exists to override it.
    assert providers.provider_for_model(model) == providers.DEFAULT_PROVIDER


def test_an_explicit_provider_overrides_the_prefix_heuristic() -> None:
    # The escape hatch for a model id the registry has never seen.
    model, provider = providers.resolve("my-custom-model", "anthropic")

    assert model == "my-custom-model"
    assert provider.name == "anthropic"


def test_a_provider_without_a_model_uses_that_providers_default() -> None:
    model, provider = providers.resolve("", "deepseek")

    assert provider.name == "deepseek"
    assert model == "deepseek-chat"


def test_neither_given_falls_back_to_openai() -> None:
    model, provider = providers.resolve("", "")

    assert provider.name == "openai"
    assert model == "gpt-4o"


def test_unknown_provider_names_fail_loudly_and_list_the_valid_ones() -> None:
    with pytest.raises(UnknownProviderError) as excinfo:
        providers.get_provider("bard")

    message = str(excinfo.value)
    assert "bard" in message
    for name in providers.PROVIDERS:
        assert name in message


@pytest.mark.parametrize("name", sorted(providers.PROVIDERS))
def test_every_provider_is_completely_specified(name: str) -> None:
    provider = providers.get_provider(name)

    assert provider.api_key_env.endswith("_API_KEY")
    assert provider.default_model
    assert provider.prefixes
    assert provider.example_models
    # Its own default must route back to itself, or `--status` would contradict itself.
    assert providers.provider_for_model(provider.default_model) == name


def test_each_provider_uses_a_distinct_api_key_variable() -> None:
    # Sharing OPENAI_API_KEY across providers would silently send the wrong credential.
    envs = [p.api_key_env for p in providers.PROVIDERS.values()]
    assert len(set(envs)) == len(envs)


def test_deepseek_is_openai_compatible_and_needs_no_extra_package() -> None:
    # This is why DeepSeek support costs zero dependencies: ChatOpenAI + a base_url.
    assert providers.get_provider("deepseek").base_url
    assert providers.get_provider("openai").base_url is None
    assert providers.get_provider("anthropic").base_url is None


def test_the_roster_names_every_provider_and_its_key() -> None:
    described = providers.describe_available()

    for provider in providers.PROVIDERS.values():
        assert provider.name in described
        assert provider.api_key_env in described
        assert provider.default_model in described
