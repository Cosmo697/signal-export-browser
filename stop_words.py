"""Stop words used by Signal Export Browser.

Kept in its own module so the main GUI file stays manageable.
"""

from __future__ import annotations

# Stop words — complete NLTK English (178 surface forms → 152 unique after
# apostrophe removal) + contractions + common English + chat filler.
#
# The tokeniser strips apostrophes (replace("'","")) before lookup, so every
# contracted form is stored WITHOUT the apostrophe (e.g. "don't" → "dont").
STOP_WORDS: frozenset[str] = frozenset(
    # ── NLTK english stop words (complete, 178 entries) ──────────────────
    # Source: nltk.corpus.stopwords.words("english")  — NLTK 3.9.1
    # Apostrophe forms listed separately after the base words.
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those "
    "am is are was were be been being have has had having "
    "do does did doing a an the and but if or because as until while "
    "of at by for with about against between through during before after "
    "above below to from up down in out on off over under "
    "again further then once here there when where why how "
    "all any both each few more most other some such "
    "no nor not only own same so than too very "
    "s t can will just don should now d ll m o re ve y "
    "ain aren couldn didn doesn hadn hasn haven isn ma "
    "mightn mustn needn shan shouldn wasn weren won wouldn "
    # NLTK words that are only present as apostrophe forms:
    #   don't  should've  aren't  couldn't  didn't  doesn't  hadn't
    #   hasn't  haven't  isn't  mightn't  mustn't  needn't  shan't
    #   shouldn't  wasn't  weren't  won't  wouldn't  it's  she's
    #   that'll  you're  you've  you'll  you'd
    # After apostrophe stripping these become:
    "dont shouldve arent couldnt didnt doesnt hadnt hasnt havent isnt "
    "mightnt mustnt neednt shant shouldnt wasnt werent wont wouldnt "
    "its shes thatll youre youve youll youd "
    # ── Extra modals (not in NLTK but fundamental) ───────────────────────
    "could would might must shall may "
    # ── Common contractions beyond NLTK ──────────────────────────────────
    "im hes theyre ive weve theyve ill hell shell well theyll "
    "cant cannot lets thats whos whats heres theres "
    "wheres whens whys hows aint "
    # ── Very-common English verbs / function words ───────────────────────
    "also got get getting gets still really actually already always "
    "never ever since much many way back new old make made makes making "
    "take took takes taking come came comes coming give gave gives giving "
    "think thought thinks thinking know knew knows knowing look looked looks "
    "looking want wanted wants wanting say said says saying tell told tells "
    "telling try tried tries trying use used uses using find found finds "
    "need needed needs needing feel felt feels feeling leave left leaves "
    "call called calls put keep keeps kept let run going gone went "
    "even though another thing things people time day good first "
    "last long great little own big small right well next around "
    "work may part something anything nothing everything someone anyone "
    "everyone two one three four five six seven eight nine ten "
    "see saw seen hear heard lot sure enough kind "
    # ── Chat / texting filler ────────────────────────────────────────────
    "like yeah yes yep yup nope nah ok okay lol haha hahaha hehe "
    "lmao lmfao omg omfg idk tbh imo imho btw brb ttyl "
    "gonna wanna gotta kinda sorta ya yea hey hi hello "
    "bye thanks thank please sorry wow oh ooh ahh hmm umm "
    "maybe probably definitely literally actually basically seriously "
    "pretty really just like well anyway "
    # ── Messaging / generic filler ───────────────────────────────────────
    "into set send sent stuff man dude cool shit "
    # ── URL fragments ────────────────────────────────────────────────────
    "http https www com org net".split()
)
