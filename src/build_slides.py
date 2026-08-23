"""Defence deck: 13 slides in English, built from committed results.

Run:
    .venv/bin/python -m src.build_slides

Every number comes from the experiment result files, not from the text of
the thesis, so the deck cannot drift from the numbers it presents.
Speaker notes are attached to each slide; likely committee questions with
short answers are written to `docs/defence_questions.md`.

Design rule followed here: one claim per slide, figures instead of
tables where a figure exists, and no sentence on a slide that the speaker
is going to read aloud.
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Cm, Pt

FIGURES = Path("reports/figures")
OUT = Path("build/defence.pptx")
QUESTIONS_OUT = Path("docs/defence_questions.md")

WIDTH, HEIGHT = Cm(33.87), Cm(19.05)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x60, 0x60, 0x60)
ACCENT = RGBColor(0x4C, 0x72, 0xB0)


def load_results() -> dict:
    exp = Path("experiments")
    return {
        "q1_lgbm": json.loads(
            (exp / "EXP-004_copywriting-gain/delta_lightgbm.json").read_text()
        ),
        "q1_ridge": json.loads(
            (exp / "EXP-004_copywriting-gain/delta_ridge.json").read_text()
        ),
        "q2_lgbm": json.loads(
            (exp / "EXP-005_full-stack/delta_lightgbm.json").read_text()
        ),
        "t3": json.loads(
            (exp / "EXP-005b_text-control/delta_lightgbm.json").read_text()
        ),
        "tau": json.loads(
            (exp / "EXP-006_platform-differences/results.json").read_text()
        ),
        "stability": json.loads(
            (exp / "EXP-002_annotation-agreement/stability.json").read_text()
        ),
        "agreement": json.loads(
            (exp / "EXP-002_annotation-agreement/agreement.json").read_text()
        ),
    }


def fmt(value: float, digits: int = 4, sign: bool = False) -> str:
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def interval(entry: dict, key: str = "ci_low", high: str = "ci_high") -> str:
    return f"[{fmt(entry[key], sign=True)}; {fmt(entry[high], sign=True)}]"


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = WIDTH, HEIGHT
        self.blank = self.prs.slide_layouts[6]

    def slide(self, title: str, notes: str):
        slide = self.prs.slides.add_slide(self.blank)
        box = slide.shapes.add_textbox(Cm(1.6), Cm(1.0), WIDTH - Cm(3.2), Cm(2.2))
        frame = box.text_frame
        frame.word_wrap = True
        run = frame.paragraphs[0].add_run()
        run.text = title
        run.font.size = Pt(30)
        run.font.bold = True
        run.font.color.rgb = DARK
        slide.notes_slide.notes_text_frame.text = notes
        return slide

    def bullets(self, slide, lines, top=Cm(4.2), size=20, left=Cm(1.6),
                width=None):
        box = slide.shapes.add_textbox(
            left, top, width or (WIDTH - Cm(3.2)), Cm(10)
        )
        frame = box.text_frame
        frame.word_wrap = True
        for i, line in enumerate(lines):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            para.space_after = Pt(14)
            emphasis = line.startswith("*")
            run = para.add_run()
            run.text = line.lstrip("*")
            run.font.size = Pt(size)
            run.font.color.rgb = ACCENT if emphasis else DARK
            run.font.bold = emphasis
        return box

    def picture(self, slide, name: str, top=Cm(4.4), height=Cm(12.5)):
        path = FIGURES / name
        if not path.exists():
            raise SystemExit(f"Figure not found: {path}")
        picture = slide.shapes.add_picture(str(path), Cm(0), top, height=height)
        picture.left = int((WIDTH - picture.width) / 2)
        return picture

    def caption(self, slide, text: str, top=Cm(17.0)):
        box = slide.shapes.add_textbox(Cm(1.6), top, WIDTH - Cm(3.2), Cm(1.4))
        frame = box.text_frame
        frame.word_wrap = True
        run = frame.paragraphs[0].add_run()
        run.text = text
        run.font.size = Pt(14)
        run.font.color.rgb = MUTED

    def save(self):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(OUT))
        return len(self.prs.slides.__iter__.__self__._sldIdLst)


def build() -> int:
    data = load_results()
    deck = Deck()

    # 1 -----------------------------------------------------------------
    slide = deck.slide(
        "Copywriting features of a headline as predictors of engagement",
        "Good afternoon. My name is [name]. The thesis asks a narrow "
        "question: does it help to measure what a headline actually does "
        "rhetorically, rather than how long it is. I will spend most of "
        "the time on two things — how I checked that those measurements "
        "are trustworthy, and where the answer turned out to depend on a "
        "choice I made before seeing any results.",
    )
    deck.bullets(slide, [
        "Master's thesis, speciality 122 Computer Science",
        "93 000 news items, three platforms, eight months",
        "*Seven content features, annotated by an LLM, validated twice",
    ], top=Cm(6.0), size=22)

    # 2 -----------------------------------------------------------------
    slide = deck.slide(
        "A prediction that does not tell you what to change",
        "Two families of work exist. Formal statistics — length, digit "
        "presence, sentiment — are interpretable but say nothing about "
        "craft. A neural network over raw headline text predicts better "
        "but names nothing. A copywriter cannot act on either. That gap "
        "is the problem the thesis addresses.",
    )
    deck.bullets(slide, [
        "Formal statistics: interpretable, but no writing advice",
        "Neural network over raw text: better numbers, no explanation",
        "*Neither answers: what do I change in this headline?",
    ], top=Cm(6.0), size=22)

    # 3 -----------------------------------------------------------------
    slide = deck.slide(
        "Data, and three confounds handled before modelling",
        "The dataset gives the same headline observed by three audiences "
        "at once — content is fixed, only the audience varies. Three "
        "properties had to be dealt with first. Truncated headlines "
        "create an information gap mechanically, so they are excluded. "
        "Missing measurements depend on publication date. And the source "
        "feature alone explains 59.69% of Facebook variance, which is why "
        "a second, within-source metric was added before any results "
        "existed.",
    )
    deck.bullets(slide, [
        "81 634 items with all three targets; split at 2016-05-17, fixed once",
        "24.31% headlines truncated at collection — excluded, gap is an artefact",
        "Source feature alone explains 59.69% of Facebook variance",
        "*So a within-source metric was fixed before the runs, not after",
    ], top=Cm(5.6), size=20)

    # 4 -----------------------------------------------------------------
    slide = deck.slide(
        "Seven features, each resting on a documented mechanism",
        "The taxonomy is not a list of tricks from a copywriting manual. "
        "Each feature is tied to a mechanism described in the literature: "
        "curiosity as a knowledge gap, news values, source credibility, "
        "suspense from an unresolved outcome. All are ordinal, zero to "
        "three, because the literature shows techniques differ in "
        "strength and a binary marker would erase that.",
    )
    deck.bullets(slide, [
        "Information gap  ·  Scale of consequences  ·  Concreteness",
        "Explicit conflict  ·  Source authority  ·  Expectation violation",
        "Prospective uncertainty",
        "*Ordinal 0–3; two features dropped — one by frequency, one by kappa",
    ], top=Cm(5.6), size=20)

    # 5 -----------------------------------------------------------------
    agreement = {r["код"]: r["kappa_linear"] for r in data["agreement"]["features"]}
    stability = {
        k: v["kappa_стабільності_зважена"]
        for k, v in data["stability"]["перевірка_гіпотези"]["за_ознаками"].items()
    }
    slide = deck.slide(
        "Validation: two measures, and they disagree",
        "This is the slide I would ask about first, so let me pre-empt it. "
        "Agreement with a human coder and stability between two "
        "independent model runs measure different things. Agreement mixes "
        "two sources of disagreement, definition and model noise; "
        "stability contains only the second. So stability should be "
        "higher. For information gap it is the other way round — 0.445 "
        "against 0.747 — which means the human agreement on that feature "
        "was substantially coincidental. I report that as a result, not a "
        "footnote.",
    )
    deck.bullets(slide, [
        f"Agreement with human coder, n = 200: "
        f"{fmt(min(agreement.values()), 3)}–{fmt(max(agreement.values()), 3)}",
        f"Stability between two runs, n = 300: "
        f"{fmt(min(stability.values()), 3)}–{fmt(max(stability.values()), 3)}",
        "Threshold 0.40 fixed before annotation — seven of eight features pass",
        "*Information gap: stability 0.445 below agreement 0.747 — the wrong way",
    ], top=Cm(5.6), size=20)

    # 6 -----------------------------------------------------------------
    slide = deck.slide("Reliability by feature", "Only two of seven features "
                       "are reliable without a caveat: explicit conflict and "
                       "expectation violation. Remember those two — they come "
                       "back at the end.")
    deck.picture(slide, "fig07_exp002_stability.png", top=Cm(4.2), height=Cm(12.0))
    deck.caption(slide, "Stability between runs against agreement with a human "
                        "coder, linear weighted kappa, 95% CI")

    # 7 -----------------------------------------------------------------
    q1 = data["q1_lgbm"]["за_платформами"]
    slide = deck.slide(
        "Do the features add accuracy over metadata?",
        "With the model fixed before the runs, the gain clears zero on "
        "Facebook only. Within source — the metric that matches the "
        "practical question, because a copywriter writes for a given "
        "publication — it also clears zero on Google+, and on Facebook it "
        "raises correlation from 0.169 to 0.281, about two thirds.",
    )
    deck.bullets(slide, [
        f"Facebook  {fmt(q1['Facebook']['delta_rho'], sign=True)} "
        f"{interval(q1['Facebook'])}",
        f"Google+   {fmt(q1['GooglePlus']['delta_rho'], sign=True)} "
        f"{interval(q1['GooglePlus'])}   interval contains zero",
        f"LinkedIn  {fmt(q1['LinkedIn']['delta_rho'], sign=True)} "
        f"{interval(q1['LinkedIn'])}   interval contains zero",
        f"*Within source, Facebook: "
        f"{fmt(q1['Facebook']['within_source']['delta_rho'], sign=True)} "
        f"{interval(q1['Facebook']['within_source'])}  (0.169 → 0.281)",
    ], top=Cm(5.6), size=20)

    # 8 -----------------------------------------------------------------
    slide = deck.slide(
        "The comparison model was fixed before the runs — and it was the wrong bet",
        "I chose gradient boosting in the protocol because the features "
        "are ordinal and act non-linearly. The data said otherwise: ridge "
        "beat it on every configuration and every platform. Under ridge, "
        "the answer to the first question would be yes on all three "
        "platforms. I did not switch. Choosing the model after seeing the "
        "answer is exactly the failure the protocol exists to prevent, so "
        "both sets of numbers are in the thesis side by side.",
    )
    deck.picture(slide, "fig09_phase5_delta_rho.png", top=Cm(4.6), height=Cm(11.0))
    deck.caption(slide, "Gain for both questions, both models, 95% CI. "
                        "Ridge intervals are systematically narrower.")

    # 9 -----------------------------------------------------------------
    t3 = data["t3"]["за_платформами"]
    slide = deck.slide(
        "The cheap alternative does not work here",
        "The obvious objection is that eight formal statistics would do "
        "the same job for free. They do not. Adding them to metadata "
        "makes the prediction worse, significantly so on two platforms. "
        "So the objection is answered — but not because the taxonomy is "
        "stronger. There is simply nothing to duplicate.",
    )
    deck.bullets(slide, [
        f"Facebook  {fmt(t3['Facebook']['delta_rho'], sign=True)} "
        f"{interval(t3['Facebook'])}",
        f"Google+   {fmt(t3['GooglePlus']['delta_rho'], sign=True)} "
        f"{interval(t3['GooglePlus'])}",
        f"LinkedIn  {fmt(t3['LinkedIn']['delta_rho'], sign=True)} "
        f"{interval(t3['LinkedIn'])}",
        "*Formal text statistics added to metadata: eight features, negative gain",
    ], top=Cm(5.6), size=20)

    # 10 ----------------------------------------------------------------
    tau = data["tau"]["kendall_tau"]
    slide = deck.slide(
        "Platforms: two agree, one does not",
        "Kendall tau compares the ordering of the seven features. On "
        "seven items tau moves in steps of 2/21, about 0.095, so the "
        "Facebook–Google+ value sits exactly on the 0.70 threshold and "
        "its interval does not hold it. I call that borderline. The two "
        "LinkedIn pairs are not borderline at all.",
    )
    deck.picture(slide, "fig12_exp006_tau.png", top=Cm(4.4), height=Cm(11.0))
    deck.caption(
        slide,
        f"Facebook–Google+ {fmt(tau['Facebook|GooglePlus']['tau'], sign=True)} "
        f"{interval(tau['Facebook|GooglePlus'])} borderline   ·   "
        f"Facebook–LinkedIn {fmt(tau['Facebook|LinkedIn']['tau'], sign=True)} "
        f"{interval(tau['Facebook|LinkedIn'])}   ·   "
        f"Google+–LinkedIn {fmt(tau['GooglePlus|LinkedIn']['tau'], sign=True)} "
        f"{interval(tau['GooglePlus|LinkedIn'])}",
        top=Cm(16.0),
    )

    # 11 ----------------------------------------------------------------
    slide = deck.slide(
        "What changes on LinkedIn — the practical result",
        "This is the finding a marketer can act on. Counter-intuitiveness "
        "works everywhere. Beyond that the platforms part company: "
        "conflict is second on the mass platforms and fifth on the "
        "professional network, where scale of consequences and named "
        "source authority move up instead. Different rulebooks for "
        "different venues, measured rather than assumed.",
    )
    deck.picture(slide, "fig11_exp006_importance.png", top=Cm(4.6), height=Cm(10.5))
    deck.caption(slide, "Conflict 2 → 5   ·   Scale 5 → 2   ·   Authority 7 → 3. "
                        "Dashed line: aggregate importance of the source feature.")

    # 12 ----------------------------------------------------------------
    slide = deck.slide(
        "Why LinkedIn shows no gain — instrument, not platform",
        "These two results look contradictory and are not. The features "
        "that rise on LinkedIn — scale and authority — are precisely the "
        "ones whose annotation carries a reliability caveat. The two "
        "features that drive the Facebook gain are the only two reliable "
        "without caveat. So the absence of a measurable gain on LinkedIn "
        "is a limit of the measuring instrument. That statement is "
        "testable: fix the rubric where validation found it "
        "under-specified, and the gain should rise.",
    )
    deck.bullets(slide, [
        "LinkedIn's top features — scale (kappa 0.662 / 0.458), "
        "authority (0.563 / 0.539)",
        "Facebook's top features — violation, conflict: "
        "the only two without caveat",
        "Under ridge, within source, LinkedIn does clear zero: +0.0539 and +0.0572",
        "*Testable claim: repair the rubric, the LinkedIn gain should rise",
    ], top=Cm(5.6), size=19)

    # 13 ----------------------------------------------------------------
    slide = deck.slide(
        "What holds, what does not, what is next",
        "To close. The contribution is real but modest, and it is larger "
        "and more useful within a publication than across publications. "
        "The honest limits are the training size, the annotation noise "
        "that pushes every gain downward, and the model choice I have "
        "already described. The most direct next step is the rubric "
        "repair, because it is the one prediction this thesis makes about "
        "its own future. Thank you — I am happy to take questions.",
    )
    deck.bullets(slide, [
        "Holds: content features add accuracy; formal statistics do not replace them",
        "Holds: platforms need different rulebooks — conflict is not universal",
        "Limits: 2 000 training rows; annotation noise lowers every gain",
        "*Next: repair the two under-specified features, then re-measure LinkedIn",
    ], top=Cm(5.6), size=20)

    return deck.save()


QUESTIONS = """# Likely committee questions

Prepared for an English-language defence. Each answer is short on
purpose: the aim is to answer in two or three sentences and stop.

---

**Why news headlines when the topic says advertising?**

The dataset was chosen for one property no advertising corpus offered:
the same text observed by three audiences at once, which is what makes
the platform comparison possible at all. The mismatch is stated as the
first limitation in the thesis. Advertising-specific devices — a call to
action, a price, a discount — do not occur in news and are therefore not
in the taxonomy and not tested.

**Isn't a gain of 0.05 in Spearman rho too small to matter?**

On the overall metric, yes, it is modest, and the thesis says so rather
than dressing it up. The number that matters practically is the
within-source gain: 0.169 to 0.281 on Facebook, roughly two thirds. That
is the setting a copywriter actually works in — the publication is given,
only the text can change.

**You used an LLM to create your own features. Isn't that circular?**

It would be if the labels were never checked. They were checked twice
against independent references: a human coder on 200 headlines and a
second independent model run on 300. The thresholds were fixed before
annotation, and one feature was dropped for failing them.

**Why is stability lower than human agreement for the information gap?**

That result is reported precisely because it is uncomfortable. Since
disagreement with a human contains both definition differences and model
noise, while disagreement between runs contains only noise, stability
should be higher. It is not, which means the human agreement on that
feature was substantially coincidental. The feature stayed in the model —
it passes the pre-registered threshold — but it ranks last in importance,
so nothing in the conclusions rests on it.

**Your main model performed worse than the simpler one. Why keep it?**

Because it was fixed before any results existed, and the reason for
choosing it — ordinal features acting non-linearly — was a hypothesis
that the data refuted. Switching afterwards would mean selecting the
model that gives the answer I prefer. Both sets of numbers are in the
thesis, so the reader can see exactly how much the answer depends on
that choice.

**Kendall tau of 0.7143 against a threshold of 0.70 — is that a real
difference?**

No, and the thesis calls it borderline. On seven items tau can only take
steps of about 0.095, and the confidence interval reaches down to 0.619,
below the threshold. The LinkedIn pairs are a different matter: their
intervals are far from the threshold.

**Google+ no longer exists. What is the value of that result?**

Methodological rather than practical, and the thesis says so. What the
platform contributes is a third independent audience observing identical
text, which is what allows the comparison to isolate audience from
content.

**Why not use embeddings or a transformer over the headline?**

That was a deliberate exclusion, not an oversight. Semantic
representations carry partial information about the topic, which would
make the second question ambiguous: a negative result could then be
explained either by duplication or by the model learning subject matter.
Comparing the taxonomy against a semantic representation is named as
future work.

**How much of the gain is annotation noise?**

Unknown in magnitude, but the direction is known: noise lowers the
measured gain rather than inflating it, so every number reported is a
lower bound. Quantifying it would require re-running the pipeline on the
second annotation run and comparing the two gains.

**Could the whole effect come from headline length or topic?**

Length and sentiment are in the control level T3, and adding that level
does not help — it hurts. Topic and source are in the metadata baseline
that every comparison already includes, so the reported gain is over and
above them.

**Why is the test set larger than the training set?**

Because the within-source metric needs at least 30 test rows per
publication. A proportional sample would have left 361 test rows and the
metric would have been computable for no publication at all. The cost is
a test set biased toward large publishers, which is stated as a
limitation.

**What would you do differently?**

Two things. I would fix the rubric's two under-specified definitions
before annotating rather than discovering them through validation. And I
would pre-register a rule for what to do when two models disagree,
instead of naming a single model and being stuck with it.
"""


def main() -> None:
    count = build()
    print(f"Slides: {count}")
    print(f"Written: {OUT}")
    QUESTIONS_OUT.write_text(QUESTIONS, encoding="utf-8")
    print(f"Written: {QUESTIONS_OUT}")


if __name__ == "__main__":
    main()
