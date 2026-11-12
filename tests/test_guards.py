"""Each rule guard with positive (should pass) and negative (should reject) examples."""

from __future__ import annotations

import pytest

from routewise.cache.guards import (
    count_negations,
    entities_guard,
    extract_numbers,
    negation_guard,
    numbers_guard,
    run_guards,
    word_order_guard,
)


@pytest.mark.parametrize("a,b", [
    ("Python 3.10 features", "Python 3.12 features"),
    ("Who won the World Cup in 2018?", "Who won the World Cup in 2022?"),
    ("What is 15% of 200?", "What is 15% of 300?"),
    ("Explain HTTP/2", "Explain HTTP/3"),
    ("Top 5 sorting algorithms", "Top 10 sorting algorithms"),
    ("Is two bigger than three?", "Is 2 bigger than 4?"),
])
def test_numbers_guard_rejects_changed_numbers(a, b):
    assert not numbers_guard(a, b)


@pytest.mark.parametrize("a,b", [
    ("Python 3.10 features", "What's new in Python 3.10?"),
    ("How do I bake bread?", "What is the way to bake bread?"),
    ("Name two primary colors", "Name 2 primary colors"),
    ("Population of 1,000,000 cities", "cities with 1000000 people"),
])
def test_numbers_guard_passes_same_numbers(a, b):
    assert numbers_guard(a, b)


def test_version_strings_kept_intact():
    assert extract_numbers("Python 3.10") != extract_numbers("Python 3.1")


def test_number_extraction_regressions():
    # Found in Phase 3 error analysis on QQP duplicates.
    assert extract_numbers("How can one live a simple life?") == set()      # "one" is a pronoun here
    assert extract_numbers("delete the first element") == set()
    assert numbers_guard("best laptop under 25000", "best laptops under 25k")
    assert numbers_guard("least 5 digit number with five figures", "least five digit number with 5 figures")


def test_quantities_are_not_entities():
    # spaCy-sm tags "Convert 5 miles" as a PERSON; found in the playground, it blocked a true paraphrase.
    assert entities_guard("Convert 5 miles to km", "How many kilometers is 5 miles?")
    # The direction swap is still rejected - by the word-order guard, whose job it is.
    r = run_guards("Convert 5 miles to km", "Convert 5 km to miles")
    assert not r.passed and r.results["word_order"] is False


def test_entity_matching_is_case_insensitive():
    assert entities_guard("Can I see who viewed my videos on Instagram?", "how do i see who viewed my video on instagram")
    assert not entities_guard("How tall is Barack Obama?", "How tall is Michelle Obama?")


@pytest.mark.parametrize("a,b", [
    ("Is coffee bad for health?", "Is coffee not bad for health?"),
    ("Can dogs eat grapes?", "Why can't dogs eat grapes?"),
    ("Which foods contain gluten?", "Which foods contain no gluten?"),
    ("Recipes with sugar", "Recipes without sugar"),
])
def test_negation_guard_rejects_polarity_flip(a, b):
    assert not negation_guard(a, b)


def test_negation_guard_passes_same_polarity():
    assert negation_guard("Why isn't my code working?", "Why does my code not work?")
    assert negation_guard("Is coffee healthy?", "Is coffee good for you?")
    assert count_negations("I don't know") == 1


@pytest.mark.parametrize("a,b", [
    ("What is the capital of India?", "What is the capital of Indonesia?"),
    ("Who founded Microsoft?", "Who founded Apple?"),
    ("How tall is Barack Obama?", "How tall is Michelle Obama?"),
    ("Best restaurants in Paris", "Best restaurants in London"),
])
def test_entity_guard_rejects_swapped_entities(a, b):
    assert not entities_guard(a, b)


@pytest.mark.parametrize("a,b", [
    ("What is the capital of India?", "Which city is India's capital?"),
    ("How do I learn Python?", "What's the best way to learn Python?"),
    ("How do I make pancakes?", "What is a good pancake recipe?"),
])
def test_entity_guard_passes_same_entities(a, b):
    assert entities_guard(a, b)


def test_word_order_guard():
    assert not word_order_guard("convert 5 miles to km", "convert 5 km to miles")
    assert not word_order_guard("Is a tomato a fruit or a vegetable?", "Is a vegetable a fruit or a tomato?")
    assert word_order_guard("convert 5 miles to km", "how many km is 5 miles")
    assert word_order_guard("same question", "same question")


def test_run_guards_reports_each_result():
    r = run_guards("What is the capital of India in 2020?", "What is the capital of Indonesia in 2021?")
    assert not r.passed
    assert set(r.failed) == {"numbers", "entities"}
    assert r.results["negation"] and r.results["word_order"]
    only_neg = run_guards("capital of India", "capital of Indonesia", ["negation"])
    assert only_neg.passed and list(only_neg.results) == ["negation"]
