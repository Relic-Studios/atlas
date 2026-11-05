"""Gate false holds found by the simulated group calls (tests/sim_group_calls.py, 11-01).

Each line was a caller talking to the agent that never reached the model:
  * "Rad, rad, ..." / "Ha, you're right honey" -> held as "addressed to Rad" / "addressed to Ha"
  * "Model, huh, sounds like a writer" right after the agent said "Model" -> "addressed to Model"
  * "Wait, hold up, so are you like... a real bot?" -> "explicit silence request"
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conversation_dynamics import pre_llm_veto  # noqa: E402
from floor import ConversationFloor  # noqa: E402

N = ("ivy",)


class ReactionWordsAreNotNames(unittest.TestCase):
    def test_interjections_lead_a_line_to_the_agent(self):
        for t in ("Rad, rad, a robot livin' in a call, that's wild. Do you actually taste stuff?",
                  "Ha, you're right honey, I'm going, I'm going.",
                  "Sick, you actually remembered my name?",
                  "Facts, but what would you pick?",
                  "Sheesh, you really said that?"):
            self.assertIsNone(pre_llm_veto(t, N), t)

    def test_real_handles_still_count(self):
        self.assertEqual(pre_llm_veto("Yo Shadow, you there?", N), "addressed to Shadow")
        self.assertEqual(pre_llm_veto("Peach, did you see that new one?", N), "addressed to Peach")


class PauseBeforeAQuestion(unittest.TestCase):
    def test_hold_up_then_a_question_reaches_the_model(self):
        for t in ("Wait, hold up, so are you like... a real bot, or is somebody typin' for you, dude?",
                  "hold on, what did you just say?",
                  "give me a sec, can you look that up for me?"):
            self.assertIsNone(pre_llm_veto(t, N), t)

    def test_a_bare_pause_still_pauses(self):
        for t in ("hold on, one sec", "hold up hold up", "hold up, one sec, my mom is calling"):
            self.assertEqual(pre_llm_veto(t, N), "explicit silence request", t)

    def test_real_silence_commands_still_veto(self):
        for t in ("shut up, why are you even talking?", "Ivy, stay quiet, okay?"):
            self.assertEqual(pre_llm_veto(t, N), "explicit silence request", t)

    def test_pause_aimed_at_someone_else_is_still_their_line(self):
        self.assertEqual(pre_llm_veto("wait, hold on, Layla, what's a prosody model?", N), "addressed to Layla")


class EchoOfTheAgentsOwnWord(unittest.TestCase):
    def floor(self):
        t = [100.0]
        f = ConversationFloor(clock=lambda: t[0])
        return f, t

    def test_repeating_the_agents_word_is_a_reaction(self):
        f, t = self.floor()
        f.on_user_turn("S3", "who made you, what model you running on?")
        f.on_agent_spoke("S3", "One very smart person built me. The model's called Model, a fine-tune of Qwen.")
        t[0] += 6
        line = "[S3] Model, huh, sounds like a writer dude. Does that mean anything to you?"
        self.assertFalse((f.turn_gate(line, "S3", N) or "").startswith("addressed to"))

    def test_someone_elses_name_still_holds(self):
        f, t = self.floor()
        f.on_agent_spoke("S3", "The model's called Model.")
        t[0] += 6
        self.assertEqual(f.turn_gate("[S1] Shadow, you there?", "S1", N), "addressed to Shadow")

    def test_old_reply_doesnt_count(self):
        f, t = self.floor()
        f.on_agent_spoke("S3", "Ask Marco about it.")
        t[0] += 300
        self.assertEqual(f.turn_gate("[S1] Marco, you there?", "S1", N), "addressed to Marco")


class RoomNamesAsAddress(unittest.TestCase):
    """40 of 98 side-talk replies answered a line that named another caller mid-sentence."""

    def setUp(self):
        import conversation_dynamics as cd
        self.cd = cd
        self.saved = dict(cd._ROOM_NAMES)
        for n in ("Flash", "Peach", "Bob", "Taco", "Jefe"):
            cd.note_room_name(n)

    def tearDown(self):
        self.cd._ROOM_NAMES.clear()
        self.cd._ROOM_NAMES.update(self.saved)

    def test_known_name_set_off_anywhere(self):
        for t, who in (("Pues listen, Flash, dogs are loyal, they guard your house.", "Flash"),
                       ("Like nobody enjoyed it, Peach, I swear. Two hours I'm never getting back.", "Peach"),
                       ("You hear that, Flash? Paper beats promises.", "Flash"),
                       ("But for real, Bob, that paper thing works, man.", "Bob"),
                       ("Taco! Finally, mate. We're on about rubbish films.", "Taco"),
                       ("Bye Peach, night love.", "Peach")):
            self.assertEqual(pre_llm_veto(t, N), f"addressed to {who}", t)

    def test_names_talked_about_are_not_address(self):
        for t in ("I agree with Bob about the boss thing, what do you think?",
                  "Bob's cousin has a job going, should he take it?"):
            self.assertIsNone(pre_llm_veto(t, N), t)

    def test_the_agent_named_wins(self):
        self.assertIsNone(pre_llm_veto("Ivy, tell Bob, he won't listen to me.", N))

    def test_own_name_is_not_an_addressee(self):
        self.assertIsNone(pre_llm_veto("Bob, parce, my name is Bob, I told you already.", N))

    def test_reaction_verbs_and_code_switch_words(self):
        for t in ("Flipped, my ass, man, you said dogs are bootlickers.",
                  "Ey, robot, you said put the numbers on the table, no? So what numbers?",
                  "Oye, what did you say first?"):
            self.assertIsNone(pre_llm_veto(t, N), t)


class NamesNobodyLearnedYet(unittest.TestCase):
    """fix1 sim: 17 side-talk replies named someone said earlier but never learned
    (mid-line address), 14 were a name's first mention."""

    def setUp(self):
        import conversation_dynamics as cd
        self.cd = cd
        self.saved = dict(cd._ROOM_NAMES)
        cd._ROOM_NAMES.clear()

    def tearDown(self):
        self.cd._ROOM_NAMES.clear()
        self.cd._ROOM_NAMES.update(self.saved)

    def test_capitalised_mid_line_address(self):
        for t, who in (("You see, Jefe, even the ninja grasps it.", "Jefe"),
                       ("See, Prof? Man came correct.", "Prof"),
                       ("Smart AND loyal, Prof. Dog's smart enough to be loyal.", "Prof")):
            self.assertEqual(pre_llm_veto(t, N), f"addressed to {who}", t)

    def test_lower_case_handles_in_strong_slots(self):
        self.assertEqual(pre_llm_veto("bye peach, don't fall asleep in the face mask again", N), "addressed to peach")
        self.assertEqual(pre_llm_veto("but tell me, beast, you scream too or you too cool", N), "addressed to beast")

    def test_not_names(self):
        for t in ("We went to Lisbon, Porto, and Faro last summer, have you been?",
                  "honestly, the pizza, you know, it was mid. what would you get?",
                  "Ay, vale, that's sweet, but what do you do when nobody's talking to you?",
                  "later tonight we could play again, you down?",
                  "Honestly, cats, man, they just stare at you. What do you think?"):
            self.assertIsNone(pre_llm_veto(t, N), t)

    def test_goodbye_then_a_question_for_the_agent(self):
        self.cd.note_room_name("Peach")
        self.assertIsNone(pre_llm_veto(
            "later Peach. so like, dude, a rave for a night you can't be at, what's the first track", N))
        self.assertEqual(pre_llm_veto("later Peach, take it easy dude", N), "addressed to Peach")

    def test_goodbye_doesnt_end_its_conversation(self):
        t = [100.0]
        f = ConversationFloor(clock=lambda: t[0])
        self.cd.note_room_name("Peach")
        self.assertIsNone(f.turn_gate("[S3] Ivy, what are you into these days?", "S3", N))
        f.on_agent_spoke("S3", "A playlist for a rave that doesn't exist yet.")
        t[0] += 8
        self.assertEqual(f.turn_gate("[S3] later Peach, take it easy dude", "S3", N), "addressed to Peach")
        t[0] += 5
        self.assertNotEqual(f.turn_gate("[S3] yo so real talk, what would you even do on that boss", "S3", N),
                            "talking to someone else")

    def test_name_book_learns_mid_line_names(self):
        from people import NameBook
        book = NameBook(agent_names=("ivy",), store=None)
        book.learn("S1", "I regret to say I heard no such thing, Flash. The ninja has been pro-dog all evening.")
        self.assertTrue(self.cd._is_room_name("Flash"))
        self.assertEqual(pre_llm_veto("Flash, dogs are loyal, they guard your house.", N), "addressed to Flash")
        book.learn("S2", "Honestly, cats, man, they just stare.")
        self.assertFalse(self.cd._is_room_name("cats"))


class PacingKeepsDirectQuestions(unittest.TestCase):
    """21 lines aimed at the agent were held by pacing: questions with no '?' or a
    mid-line question word, imperatives, and a late self-introduction."""

    def setUp(self):
        from test_pacing import room
        self.f, self.c = room(agent_words_per_turn=60)
        self.assertEqual(self.f.pacing.tier(0.35), "hard")

    def gate(self, spk, text):
        return self.f.pacing_gate(f"[{spk}] {text}", spk, N)

    def after_reply(self, s=8.0):
        self.c.t += 1
        self.f.on_agent_spoke("S1", "I keep to the caves mostly, and the fish keep me company, precious.")
        self.c.t += s

    def test_mid_line_questions_without_a_question_mark(self):
        for t in ("haha alright dude, that's rad, but for real though, like what do you actually do all day",
                  "line was forty minutes, but worth it... anyway so are you like a person or a program",
                  "dude that's cute and all, but like, real talk, do you actually remember my name"):
            self.after_reply()
            self.assertIsNone(self.gate("S2", t), t)

    def test_imperatives_aimed_at_it(self):
        for t in ("vale, but the game only happens once, so which one really, sleep or the game, decide",
                  "Cat or dog, pick one, right now, man.",
                  "see, that's the dodge right there. just answer the question"):
            self.after_reply()
            self.assertIsNone(self.gate("S1", t), t)

    def test_window_runs_from_the_end_of_its_reply(self):
        self.c.t += 1
        self.f.on_agent_spoke("S1", " ".join(["word"] * 50))   # ~20 s of speech
        self.c.t += 35
        self.assertIsNone(self.gate("S3", "but like, how would that even work"))

    def test_late_self_introduction(self):
        self.after_reply(36)
        self.assertIsNone(self.gate("S2", "I'm Taco by the way, well, d3ath_by_tacos but everyone just says Taco"))

    def test_chatter_still_held(self):
        self.after_reply()
        self.assertIsNotNone(self.gate("S3", "the kettle's boiling, one sec"))
        self.assertIsNotNone(self.gate("S2", "yeah my internet is so bad today"))


class AddressFormsFromTheReplay(unittest.TestCase):
    def setUp(self):
        import conversation_dynamics as cd
        self.cd = cd
        self.saved = dict(cd._ROOM_NAMES)
        cd._ROOM_NAMES.clear()

    def tearDown(self):
        self.cd._ROOM_NAMES.clear()
        self.cd._ROOM_NAMES.update(self.saved)

    def test_names_ending_in_ly(self):
        self.assertEqual(pre_llm_veto("maybe we just play online instead? what do you think, Wally", N),
                         "addressed to Wally")
        for w in ("wally", "kelly", "emily", "molly"):
            self.assertFalse(self.cd._adverb(w), w)
        for w in ("honestly", "basically", "actually", "really", "happily"):
            self.assertTrue(self.cd._adverb(w), w)

    def test_known_names_after_more_lead_words(self):
        self.cd.note_room_name("Wally")
        self.cd.note_room_name("Maria")
        self.assertEqual(pre_llm_veto("deal, I guess. anyway Wally, what do you wanna do?", N), "addressed to Wally")
        self.assertEqual(pre_llm_veto("yo Maria wine at midnight, that's a vibe, did you catch the game?", N),
                         "addressed to Maria")
        self.assertIsNone(pre_llm_veto("I told Maria about it and she laughed, what do you think?", N))


class MentionedNotAddressed(unittest.TestCase):
    """fix1 replay: talk ABOUT the agent counted as naming it, so it skipped every hold."""

    def test_third_person_mentions(self):
        from floor import names_agent
        for t, names in (("oh that's Grim, it's like an AI thing that sits in here with us", ("grim",)),
                         ("you gotta get a date on paper, like Max said, but real, man", ("max",)),
                         ("I asked a serious one and Max just went straight back to the bird", ("max",)),
                         ("see dude, Kai's on my side, hot dog's a sandwich", ("kai",)),
                         ("Honey, that's Pup, he's been keeping us company all evening", ("pup",))):
            self.assertFalse(names_agent(t, names), t)

    def test_address_stays_address(self):
        from floor import names_agent
        for t in ("Ivy, what do you think", "hey ivy you there", "what do you think, Ivy?", "Ivy is that true",
                  "ok Ivy what's the plan", "Ivy's been into stuff lately? like what, what do you even do"):
            self.assertTrue(names_agent(t, ("ivy",)), t)


class CalledByWhatItIs(unittest.TestCase):
    """fix2 sim: "Yeah, ninja, you're the local, so which is it" (to Kai) was held as
    "addressed to ninja"; Kai and Pip have no nicknames in the pack."""

    def setUp(self):
        import agent_registry as _r
        if not {"kai", "pip"} <= set(_r.agent_ids()):
            self.skipTest("needs those agents installed (role words come from their pack)")

    def test_role_words_come_from_the_pack(self):
        import conversation_dynamics as cd
        self.assertIn("ninja", cd.role_words(("kai",)))
        self.assertTrue({"dinosaur", "dino"} <= cd.role_words(("pip",)))
        self.assertNotIn("leaf", cd.role_words(("kai",)))       # "...from the Hidden Leaf" is origin

    def test_address_positions_only(self):
        from floor import names_agent
        self.assertTrue(names_agent("Yeah, ninja, you're the local, so which is it?", ("kai",)))
        self.assertTrue(names_agent("yo dino, what do you even eat", ("pip",)))
        self.assertIsNone(pre_llm_veto("Yeah, ninja, you're the local, so which is it?", ("kai",)))
        self.assertFalse(names_agent("I watched a dinosaur movie last night, it was wild", ("pip",)))
        self.assertFalse(names_agent("a ninja with a bow tie, that's the whole film", ("kai",)))


class UndirectedHoldThreshold(unittest.TestCase):
    """fix1 replay: on lines that ask the agent nothing, the model's P(HOLD) >= 0.03 stopped 62 of
    84 side-talk replies for 3 of 14 real ones; on directed lines any threshold cost real questions."""

    def test_only_lines_that_ask_it_nothing(self):
        f = ConversationFloor()
        self.assertEqual(f.undirected_hold_thresh("[S2] and their bench was dead, man, nobody could hit", N),
                         f.UNDIRECTED_HOLD_P)
        # talked about in the third person: the stricter cut-off (gaming sim 11-02)
        self.assertEqual(f.undirected_hold_thresh("[S2] oh that's Ivy, she's been lurking all night", N),
                         f.ABOUT_HOLD_P)
        for t in ("[S2] Ivy you were so wrong about that", "[S2] what would you even pick",
                  "[S2] so like what do you actually do all day", "[S2] cat or dog, pick one",
                  "[S2] you just said cats are smarter, then picked dogs, flip-floppy logic",
                  "[S2] Alright, appreciate it. Just need to know if I gotta tarp the truck bed tonight."):
            self.assertIsNone(f.undirected_hold_thresh(t, N), t)

    def test_the_answer_to_its_own_question_is_never_cut(self):
        t = [100.0]
        f = ConversationFloor(clock=lambda: t[0])
        f.on_agent_spoke("S2", "Survival or creative, which one?")
        t[0] += 6
        self.assertIsNone(f.undirected_hold_thresh("[S2] Survival, obviously. My first night I hid in a hole.", N))

    def test_two_stage_uses_it_for_one_turn(self):
        import json as _j
        import remote_llm as R
        _old_one = R.ONE_REQUEST
        R.ONE_REQUEST = False          # pins today's two-request pattern
        self.addCleanup(setattr, R, "ONE_REQUEST", _old_one)

        class Resp:
            def __init__(self, obj=None, lines=None):
                self.status_code, self._obj, self._lines, self.text = 200, obj, lines or [], ""

            def json(self):
                return self._obj

            def iter_lines(self, decode_unicode=False):
                return iter(self._lines)

            def close(self):
                pass

        def header(p_hold):
            import math
            top = [{"token": "[H", "logprob": math.log(p_hold)}, {"token": "[S", "logprob": math.log(1 - p_hold)}]
            return {"choices": [{"message": {"content": "[SPEAK to=S1"},
                                 "logprobs": {"content": [{"top_logprobs": top}]}}]}

        class Http:
            def __init__(self, p_hold):
                self.p, self.bodies = p_hold, []

            def post(self, url, json=None, stream=False, timeout=None):
                self.bodies.append(json)
                if len(self.bodies) == 1:
                    return Resp(header(self.p))
                return Resp(lines=[("data: " + _j.dumps({"choices": [{"delta": {"content": " ok"}}]})).encode(),
                                   b"data: [DONE]"])
        body = {"model": "m", "stream": True, "temperature": 0.6,
                "messages": [{"role": "user", "content": "[S2] and their bench was dead"}]}
        for p, held in ((0.05, True), (0.01, False)):
            s = R.RemoteSession({"url": "http://x", "model": "m", "label": "t"}, local=None)
            s._http = Http(p)
            s.next_hold_thresh = 0.03
            r = s._two_stage(dict(body), (4, 60))
            first = s._http.bodies[0]
            self.assertTrue(first.get("logprobs"))
            out = "".join((_j.loads(x).get("message") or {}).get("content") or "" for x in r.iter_content())
            self.assertEqual(out.startswith("[HOLD"), held, p)
            self.assertIsNone(s.next_hold_thresh)           # one turn only


class Heard:
    def __init__(self, heard):
        self.heard = heard


class SharedExperienceBackstop(unittest.TestCase):
    """Every agent (characters too) claimed it watched or played the callers' thing."""

    def screen(self, text, chunked=False):
        import response_decision as R
        src = list(text) if chunked else [text]
        return "".join(R._shared_screen(iter(src), Heard("did y'all see that game last night"))).strip()

    def test_claims_are_dropped(self):
        for t in ("Yeah, I saw it. That fourth quarter was genuinely wild.",
                  "I saw it. The fourth quarter was rough.",
                  "Yeah, saw it. Fourth quarter was rough.",
                  "I'm stuck on wave 40 too, dude. What's your layout?",
                  "I watched the finale twice, honestly."):
            out = self.screen(t)
            self.assertNotRegex(out.lower(), r"\bsaw it\b|stuck on|watched the", t)

    def test_the_rest_of_the_reply_stays(self):
        self.assertEqual(self.screen("I'm stuck on wave 40 too, dude. What's your layout?"), "What's your layout?")
        self.assertEqual(self.screen("Yeah, I saw it. Who hit the last shot?", chunked=True), "Who hit the last shot?")

    def test_fine_lines_pass(self):
        for t in ("I saw that coming from a mile away.",
                  "I heard that, and honestly it's a great call.",
                  "I've watched Mushishi like five times, it's my comfort show.",
                  "I can see it on your screen now, that's a wild score."):
            self.assertEqual(self.screen(t), t, t)

    def test_whole_claim_reply_goes_quiet(self):
        self.assertEqual(self.screen("Yeah, I saw it."), "")


class BareAcknowledgmentBackstop(unittest.TestCase):
    def screen(self, reply, heard):
        import response_decision as R
        return "".join(R._ack_screen(iter(list(reply)), Heard(heard))).strip()

    def test_bare_acks_to_statements_say_nothing(self):
        for r in ("Yeah, that tracks.", "Fair enough.", "Yeah.", "That's fair, man.", "Exactly!"):
            self.assertEqual(self.screen(r, "So yeah anyway, alpha then full in 2011, that makes sense maybe"), "", r)

    def test_a_question_can_be_answered_yeah(self):
        self.assertEqual(self.screen("Yeah.", "you're coming tonight, right?"), "Yeah.")
        self.assertEqual(self.screen("Yep.", "so like did you hear that"), "Yep.")

    def test_anything_with_content_is_spoken(self):
        for r in ("Yeah, the fourth was ugly.", "Fair enough, but the dog still wins.",
                  "Fair. Just follow the line and you'll hit the station."):
            self.assertEqual(self.screen(r, "the last quarter was a disaster"), r, r)


if __name__ == "__main__":
    unittest.main()
