"""Self-check for the local contacts book. Run: python test_contacts.py"""
import tempfile
from pathlib import Path

import agent


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        agent.CONTACTS_FILE = Path(d) / "contacts.json"

        # save with aliases; number is normalized to digits (drops +, spaces, dashes)
        agent.save_contact.invoke({"name": "Dad", "number": "+507 687-8965", "aliases": ["papa", "el viejo"]})
        assert agent.lookup_contact.invoke({"name": "dad"}) == "5076878965"
        assert agent.lookup_contact.invoke({"name": "PAPA"}) == "5076878965"      # alias, case-insensitive
        assert agent.lookup_contact.invoke({"name": "my papa"}) == "5076878965"   # partial match

        # unknown name lists what's known instead of guessing
        miss = agent.lookup_contact.invoke({"name": "nobody"})
        assert "No contact" in miss and "dad" in miss, miss

        # ambiguous partial across two different numbers → asks the user
        agent.save_contact.invoke({"name": "Ana", "number": "50761112222"})
        agent.save_contact.invoke({"name": "Andres", "number": "50763334444"})
        amb = agent.lookup_contact.invoke({"name": "An"})
        assert "Multiple matches" in amb, amb

    print("OK")


if __name__ == "__main__":
    main()
