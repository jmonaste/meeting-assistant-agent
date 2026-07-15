"""Language support: the output-language rule and the bilingual cue scan."""

from __future__ import annotations

from pathlib import Path

from meeting_assistant.config import load_settings
from meeting_assistant.extractors import scan_signals
from meeting_assistant.graph.builder import run_pipeline
from meeting_assistant.graph.nodes import language_instruction
from meeting_assistant.ingest.loader import build_inventory

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------- instruction

def test_language_instruction_auto_follows_transcript():
    note = language_instruction("auto")
    assert "SAME language" in note
    assert language_instruction("") == note
    assert language_instruction("AUTO") == note


def test_language_instruction_known_code_maps_to_name():
    assert "Spanish" in language_instruction("es")
    assert "English" in language_instruction("en")
    assert "Spanish" in language_instruction("ES")  # case-insensitive


def test_language_instruction_unknown_value_passes_through():
    assert "klingon" in language_instruction("klingon")


def test_quotes_never_translated_in_both_modes():
    assert "quotes exactly as spoken" in language_instruction("auto")
    assert "untranslated" in language_instruction("es")


# ------------------------------------------------------------- prompt wiring

def test_pipeline_prompts_carry_the_language_rule(fake_llm, settings, sample_inventory, tmp_path):
    settings.language = "es"
    run_pipeline(settings=settings, inventory=sample_inventory, llm=fake_llm, cache_base=tmp_path)
    generating = [s for s in fake_llm.systems if "OUTPUT LANGUAGE" in s]
    assert generating, "no prompt carried the language rule"
    assert all("Spanish" in s for s in generating)
    # Every structured call in the pipeline generates user-facing text.
    assert len(generating) == len(fake_llm.systems)


# --------------------------------------------------------- bilingual cue scan

def test_cue_scan_finds_spanish_signals(settings):
    text = (FIXTURES / "sample-reunion-es.txt").read_text(encoding="utf-8")
    inv = build_inventory(text, settings, source_file=str(FIXTURES / "sample-reunion-es.txt"))
    assert inv.source_format == "labeled"
    assert {sp.name for sp in inv.speakers} == {"Ana", "Luis", "Marta", "Diego"}

    sig = scan_signals(inv)
    assert sig["action_items"]      # "Me encargo", "Hay que", "envía el plan"
    assert sig["decisions"]         # "De acuerdo", "Acordado", "Decidido"
    assert sig["key_dates"]         # "el viernes", "15 de agosto", "miércoles"
    assert sig["risks"]             # "bloqueado", "riesgo", "retrasa"
    assert sig["key_facts"]         # "4000 euros", "30 por ciento"
    assert sig["open_questions"]    # inverted "¿" question marks


def test_cue_scan_inverted_question_mark_mid_line(settings):
    # A Spanish question embedded mid-line (no trailing "?") is still detected.
    inv = build_inventory(
        "Ana: Tengo una duda: ¿incluimos el panel en la 2.0 o no? Ya veremos.",
        load_settings(MEETING_USE_CACHE=False),
    )
    sig = scan_signals(inv)
    assert sig.get("open_questions")


def test_cue_scan_still_finds_english_signals(sample_inventory):
    # The bilingual patterns must not regress the English scan.
    sig = scan_signals(sample_inventory)
    assert sig["action_items"]
    assert sig["decisions"]
    assert sig["key_dates"]
