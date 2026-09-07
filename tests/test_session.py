from langgraph.store.memory import InMemoryStore

from session import resolve_thread_id


def test_mints_and_saves_a_new_id_on_first_run() -> None:
    store = InMemoryStore()

    thread_id, was_resumed = resolve_thread_id(store)

    assert thread_id
    assert was_resumed is False


def test_resumes_the_previously_saved_id() -> None:
    store = InMemoryStore()
    first_id, _ = resolve_thread_id(store)

    second_id, was_resumed = resolve_thread_id(store)

    assert second_id == first_id
    assert was_resumed is True


def test_start_new_mints_a_fresh_id_and_overwrites_the_saved_one() -> None:
    store = InMemoryStore()
    first_id, _ = resolve_thread_id(store)

    second_id, was_resumed = resolve_thread_id(store, start_new=True)
    third_id, third_was_resumed = resolve_thread_id(store)

    assert second_id != first_id
    assert was_resumed is False
    # the forced-new id is now the one a later un-forced call resumes
    assert third_id == second_id
    assert third_was_resumed is True
