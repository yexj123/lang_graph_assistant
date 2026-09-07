"""Which LLM provider backs a given model name.

Pure registry: no langchain imports, no network, no config. `config.get_model()` reads
this to decide which client class to build, and everything else (the CLI banner, the
eval's judge-independence check, `--status`) can ask questions about a model without
constructing one.

Adding a provider means adding an entry to PROVIDERS and a branch in
`config._build_chat_model()` — nothing else in the codebase should learn provider names.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    name: str
    api_key_env: str
    default_model: str
    # Model-id prefixes that identify this provider, so `OPENAI_MODEL=claude-sonnet-5`
    # routes to Anthropic without the user also setting LLM_PROVIDER.
    prefixes: tuple[str, ...]
    # OpenAI-compatible providers reuse ChatOpenAI with a different endpoint.
    base_url: str | None = None
    example_models: tuple[str, ...] = field(default_factory=tuple)


PROVIDERS: dict[str, Provider] = {
    "openai": Provider(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o",
        prefixes=("gpt-", "o1", "o3", "o4", "chatgpt"),
        example_models=("gpt-4o", "gpt-4o-mini", "gpt-4.1"),
    ),
    "anthropic": Provider(
        name="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-5",
        prefixes=("claude",),
        example_models=("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"),
    ),
    "deepseek": Provider(
        name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        prefixes=("deepseek",),
        # DeepSeek serves an OpenAI-compatible API, so it needs no extra dependency -
        # ChatOpenAI with a different base_url is the whole integration.
        base_url="https://api.deepseek.com",
        example_models=("deepseek-chat", "deepseek-reasoner"),
    ),
}

DEFAULT_PROVIDER = "openai"


class UnknownProviderError(ValueError):
    """Raised for a provider name that is not registered."""


def provider_for_model(model_name: str) -> str:
    """Infer the provider from a model id, falling back to the default.

    Prefix matching means `OPENAI_MODEL=claude-sonnet-5` just works. It is a heuristic,
    which is why LLM_PROVIDER exists to override it explicitly.
    """
    lowered = (model_name or "").strip().lower()
    for provider in PROVIDERS.values():
        if any(lowered.startswith(prefix) for prefix in provider.prefixes):
            return provider.name
    return DEFAULT_PROVIDER


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name.strip().lower()]
    except KeyError:
        raise UnknownProviderError(
            f"Unknown provider {name!r}. Known providers: {', '.join(sorted(PROVIDERS))}."
        ) from None


def resolve(model_name: str = "", provider_name: str = "") -> tuple[str, Provider]:
    """Settle on a concrete (model id, provider) pair from whatever the user supplied.

    Both may be empty: an explicit provider with no model uses that provider's default,
    a model with no provider is inferred by prefix, and neither falls back to OpenAI.
    """
    if provider_name:
        provider = get_provider(provider_name)
        return (model_name or provider.default_model), provider

    if model_name:
        return model_name, get_provider(provider_for_model(model_name))

    provider = get_provider(DEFAULT_PROVIDER)
    return provider.default_model, provider


def describe_available() -> str:
    """Human-readable roster for `--status` and error messages."""
    lines = []
    for provider in PROVIDERS.values():
        models = ", ".join(provider.example_models)
        lines.append(
            f"  {provider.name:<10} key={provider.api_key_env:<18} default={provider.default_model}\n"
            f"             e.g. {models}"
        )
    return "\n".join(lines)
