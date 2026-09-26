"""End-to-end REPL tests for c_neural_server's neural memory path.

Regression guards for the LC-001 failure fixes: episode capacity (>16),
adversarial-question refusal, in-distribution recall, and persistence.
Drives the real binary over stdin/stdout, exactly like the LoCoMo driver.
"""
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from eval_locomo_neural import Server, classify, episode_count  # noqa: E402

SERVER = ROOT / "build/c_neural_server"
GGUF = ROOT / "models/bitcpm4-0.5b-tq2_0.gguf"
MODEL = ROOT / "build/neural-c-weights/model.bnmodel"


class ServerArgs:
    server = str(SERVER)
    gguf = str(GGUF)
    memory_model = str(MODEL)


@unittest.skipUnless(SERVER.exists() and GGUF.exists() and MODEL.exists(),
                     "server binary, GGUF, or bnmodel missing")
class NeuralServerMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def ask(self, server, question):
        reply, err = server.message(question)
        return reply, err, classify(err)

    def test_capacity_beyond_16(self):
        server = Server(ServerArgs, self.dir)
        server.start()
        try:
            for i in range(30):
                server.message(f"Person{i} lives in City{i}.")
            status = server.command("/status")
            episodes = episode_count(status)
            self.assertIsNotNone(episodes)
            self.assertGreater(episodes, 16, "/status must report >16 episodes")
        finally:
            server.quit()

    def test_unbounded_store_and_long_episode(self):
        """Memory content is unbounded: 600 episodes all stored, and a
        multi-kilobyte episode keeps its fact recallable (feature window
        covers the first ~500 tokens, so the fact leads the padding)."""
        server = Server(ServerArgs, self.dir)
        server.start()
        try:
            for i in range(600):
                server.message(f"Fact number {i} is about topic {i}.")
            status = server.command("/status")
            self.assertEqual(episode_count(status), 600)

            fact = ("Caius currently lives in Aachen. "
                    "Darya currently works in Bilbao.")
            server.message(fact + " Background: " + "p" * 3000 + ".")
            status = server.command("/status")
            self.assertEqual(episode_count(status), 601)

            reply, err, info = self.ask(server, "Where does Caius live now?")
            self.assertTrue(info["activated"],
                            f"long-episode recall: {err[-300:]}")
            self.assertEqual(reply.strip(), "Aachen")
        finally:
            server.quit()

    def test_in_distribution_recall(self):
        """Recall on the JB training form (two-fact episode, in-distribution
        subject-attribute pair). OOD pairs are a documented limitation
        (PAPER.md / PLAN.md binding wall), not a test target."""
        server = Server(ServerArgs, self.dir)
        server.start()
        try:
            server.message("Caius currently lives in Aachen. "
                           "Darya currently works in Bilbao.")

            reply, err, info = self.ask(server, "Where does Caius live now?")
            self.assertTrue(info["activated"], f"Caius query should activate: {err[-300:]}")
            self.assertEqual(reply.strip(), "Aachen")
        finally:
            server.quit()

    def test_multi_episode_no_wrong_value(self):
        """Safety invariant: with more than one episode stored and no episode
        selection (EC-001 gate failed), the server must either recall the
        correct value or refuse — never return another subject's value."""
        server = Server(ServerArgs, self.dir)
        server.start()
        try:
            server.message("Morgan lives in Lima.")
            server.message("Caius works in Aachen.")

            reply, err, info = self.ask(server, "Where does Morgan live now?")
            if info["activated"]:
                self.assertEqual(reply.strip(), "Lima",
                                 f"activated recall must be correct: {err[-300:]}")
            # non-activated (refusal) is acceptable; wrong value is not

            reply, err, info = self.ask(server, "Where does Caius work now?")
            if info["activated"]:
                self.assertEqual(reply.strip(), "Aachen",
                                 f"activated recall must be correct: {err[-300:]}")
        finally:
            server.quit()

    def test_multi_episode_recall_non_last(self):
        """Usable-model acceptance (EC-005 A+B): with several episodes stored,
        a query about a NON-LAST episode's subject must recall that episode's
        value — episode selection, not last-episode fallback. All three pairs
        are JB-001 training items (in-distribution by construction)."""
        server = Server(ServerArgs, self.dir)
        server.start()
        try:
            server.message("Caius currently lives in Aachen. "
                           "Darya currently works in Bilbao.")
            server.message("Tavian currently lives in Siena. "
                           "Vera currently works in Tartu.")
            server.message("Kellan currently lives in Kiel. "
                           "Liora currently works in Lugano.")

            reply, err, info = self.ask(server, "Where does Caius live now?")
            self.assertTrue(info["activated"],
                            f"first-episode query must activate: {err[-300:]}")
            self.assertEqual(reply.strip(), "Aachen")

            reply, err, info = self.ask(server, "Where does Tavian live now?")
            self.assertTrue(info["activated"],
                            f"middle-episode query must activate: {err[-300:]}")
            self.assertEqual(reply.strip(), "Siena")
        finally:
            server.quit()

    def test_dialogue_form_recall(self):
        """EC-006: colloquial episodes (speaker-prefixed turns) + natural
        question forms must recall correctly — the form-diversity goal."""
        server = Server(ServerArgs, self.dir)
        server.start()
        try:
            server.message("Krifell: I'm based in Frtorp these days, really "
                           "enjoying it. Kriwick: work update — I'm working "
                           "in Frdal now.")
            server.message("Krifell: I'm based in Frtorp these days, really "
                           "enjoying it. Kriwick: work update — I'm working "
                           "in Frdal now. Krifell: the new job is in Frdal, "
                           "so that's where I am.")

            reply, err, info = self.ask(server, "Where is Krifell living these days?")
            self.assertTrue(info["activated"],
                            f"dialogue query must activate: {err[-300:]}")
            self.assertEqual(reply.strip(), "Frtorp")
        finally:
            server.quit()

    def test_paraphrase_form_recall(self):
        """EC-007: 3B-paraphrased episodes (teacher-verified spans) must
        recall — the real-dialogue-diversity goal."""
        server = Server(ServerArgs, self.dir)
        server.start()
        try:
            server.message("Zcino: Hey friend! Just wanted to share a quick "
                           "update on Yunas. It's been a beautiful day with "
                           "the sun Zcella2: So, you're saying you're "
                           "currently working in Yusund?")
            reply, err, info = self.ask(server, "Where does Zcino currently reside?")
            self.assertTrue(info["activated"],
                            f"paraphrase query must activate: {err[-300:]}")
            self.assertEqual(reply.strip(), "Yusund")
        finally:
            server.quit()

    def test_adversarial_question_not_activated(self):
        server = Server(ServerArgs, self.dir)
        server.start()
        try:
            server.message("Morgan lives in Lima.")
            reply, err, info = self.ask(server, "Where does Norbert live now?")
            self.assertFalse(info["activated"],
                             "unknown subject must not trigger value recall")
        finally:
            server.quit()

    def test_persistence_roundtrip(self):
        state = self.dir / "state.bnstate"
        server = Server(ServerArgs, self.dir)
        server.start()
        try:
            server.message("Caius currently lives in Aachen. "
                           "Darya currently works in Bilbao.")
            server.command(f"/save {state}")
        finally:
            server.quit()

        server = Server(ServerArgs, self.dir)
        server.start()
        try:
            server.command(f"/load {state}")
            status = server.command("/status")
            self.assertEqual(episode_count(status), 1)
            reply, err, info = self.ask(server, "Where does Caius live now?")
            self.assertTrue(info["activated"], f"recall after reload: {err[-300:]}")
            self.assertEqual(reply.strip(), "Aachen")
        finally:
            server.quit()


if __name__ == "__main__":
    unittest.main()
