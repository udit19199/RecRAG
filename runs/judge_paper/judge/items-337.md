Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=77edc3a6772442f3 ===
Gold sentences and facts:
Sentence 0: The Cowboys opened Week 13 by rallying to beat the Seahawks on Thursday night, keeping the heat on the losing Eagles ahead of them before hosting that team in Dallas in Week 14.
  Fact 0.0: The Cowboys opened Week 13 by rallying to beat the Seahawks on Thursday night.
  Fact 0.1: The Cowboys kept the heat on the Eagles.
  Fact 0.2: The Eagles lost their game.
  Fact 0.3: The Eagles were ahead of the Cowboys.
  Fact 0.4: The Cowboys hosted the Eagles in Dallas in Week 14.
Sentence 1: Philadelphia rolled to a 31-17 win over Miami on "Sunday Night Football," keeping the Dolphins' offense off the scoreboard in the second half and playing smart football every step of the way down the stretch.
  Fact 1.0: Philadelphia beat Miami 31-17.
  Fact 1.1: Philadelphia's win over Miami was on "Sunday Night Football".
  Fact 1.2: Philadelphia rolled to the win.
  Fact 1.3: Philadelphia kept the Dolphins' offense off the scoreboard in the second half.
  Fact 1.4: Philadelphia played smart football every step of the way down the stretch.

Retrieved items:
<item 1>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
He gets Philadelphia a first down.

8:23 p.m. — The Eagles will start the game with the ball, and we are underway!

8:18 p.m. — As many have pointed out, Tua Tagovailoa and Jalen Hurts were once teammates and competitors at Alabama. Both have something to prove tonight,
considering Hurts hasn't looked like himself for much of this season.

8:10 p.m. — The Dolphins can open up a two-game AFC East lead on the Bills with a win tonight. Buffalo was upset by the Patriots earlier today.

8:03 p.m. — Lincoln Financial Field has a new look for Kelly Green day.

Lincoln Financial Field, with a new logo and wordmark for this stadium: pic.twitter.com/vHEm7uBDgi — Zach Berman (@ZBerm) October 22, 2023

7:55 p.m. — Oh yeah, Julio Jones is back in the NFL.
</item 1>

<item 2>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
Game: Dolphins vs. Eagles

Dolphins vs. Eagles Date: Sunday, Oct. 22

Sunday, Oct. 22 TV channel: NBC

NBC Live stream: Peacock | NBCSports.com | Fubo (U.S.) | DAZN (Canada)

Dolphins vs. Eagles can be watched on NBC, which has long held the broadcast rights to "Sunday Night Football."

Mike Tirico (play-by-play) and Cris Collinsworth (color analyst) will be on the call from Levi's Stadium while Melissa Stark will serve as the sideline
reporter. The crew is in its second season together after Tirico replaced play-by-play legend Al Michaels in 2022 while Stark took over for Michele Tafoya.

Fans can also stream the game via Peacock or NBCSports.com with a subscription, along with Fubo, which offers a free trial.

Viewers in Canada can stream every NFL game via DAZN.

Dolphins vs.
</item 2>

<item 3>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
After the Eagles' defense clamped down on Tagovailoa and the Dolphins, the offense pieced together a touchdown drive spanning nearly seven minutes to ice the
game.

A.J. Brown overcame a slow start and shined for the Eagles, posting 137 yards and a touchdown in the win. Brown has now ripped off five consecutive games of at
least 125 yards and seems to be on his way to a career-best season.

Hurts, Brown and the Eagles have higher goals than just individual success, and a win over one of the AFC's best teams is one way to make a statement.
Philadelphia moves to 6-1, while the Dolphins miss a chance to open up a two-game lead in the AFC East and fall to 5-2 ahead of a matchup with the Patriots.

MORE: Latest injury updates on Dolphins WR Jaylen Waddle

The Sporting News tracked live scoring updates and highlights from "Sunday Night Football" between the Dolphins and Eagles. Check out all of the key moments you
might have missed.
</item 3>

<item 4>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
Deion Sanders once said, “If you look good, you feel good, and if you feel good, you play good." The Eagles looked good on Sunday night, donning Kelly Green
jerseys and helmets for their matchup with the Dolphins.

And their play absolutely matched the look.

Philadelphia rolled to a 31-17 win over Miami on "Sunday Night Football," keeping the Dolphins' offense off the scoreboard in the second half and playing smart
football every step of the way down the stretch.

The Dolphins quickly made up a 17-3 deficit, scoring a touchdown just before halftime and finding the end zone again in the third quarter when Jerome Baker
intercepted a deflected pass from Jalen Hurts for a pick-six.

MORE: How Mike McDaniel overcame alcoholism to become rising NFL coach

The defining series of the game came right after the pick-six when Hurts shrugged off the turnover and led a surgical touchdown drive.
</item 4>

<item 5>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
Eagles 24, Dolphins 17

10:35 p.m. TOUCHDOWN — What an answer from the Eagles! A.J. Brown breaks a tackle and goes right into the end zone to put the Eagles back in front. It's 24-17.

10:34 p.m. — Goedert comes through with a huge catch to extend the Eagles' drive and enter the red zone.

10:33 p.m. — The Eagles are getting chunks of yards early on this drive, including a 25-yard catch by DeVonta Smith as they barrel into Dolphins territory.

10:30 p.m. — Jevon Holland and David Long were both injured on the kickoff, resulting in a bit of a delay. They both were able to walk off.

Dolphins 17, Eagles 17

10:23 p.m. TOUCHDOWN — Watch out! Kader Kohou deflects a pass from Hurts and it's returned by Jerome Baker for a touchdown.
</item 5>

<item 6>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
It'll be a field goal try for Jake Elliott.

8:30 p.m. — Jalen Hurts is celebrating National Tight Ends Day by hitting Dallas Goedert on back-to-back passes for a total of 45 yards. The Eagles are into the
red zone.

Fitting that the first big play of the game goes to Dallas Goedert on #NationalTightEndsDay



📺: #MIAvsPHI on NBC

📱: Stream on #NFLPlus https://t.co/0rIJPFKW6L pic.twitter.com/DPfTTVQbnz — NFL (@NFL) October 23, 2023

8:28 p.m. — Swift has five touches already. No one else has seen the ball.

8:26 p.m. — The Eagles are riding D'Andre Swift early, as the running back is up to three carries through three snaps. He gets Philadelphia a first down.

8:23 p.m.
</item 6>

<item 7>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
9:15 p.m. — Julio Jones is on the board with his first catch as an Eagle. It's a three-yard gain for the future Hall of Famer.

9:12 p.m. — The Eagles have the Dolphins' offense uncomfortable so far. Miami is punting after another Tagovailoa incompletion, and this will be a big moment
for the Dolphins' defense.

Eagles 10, Dolphins 3

9:06 p.m. TOUCHDOWN — Happy National Tight Ends Day! Dallas Goedert with a great catch-and-run all the way into the end zone to give the Eagles the lead.

9:05 p.m. — Christian Wilkins gets called for roughing the passer, pushing the Eagles right to the edge of the red zone.

9:03 p.m. — On 4th & 1, you already know what the play is. Hurts gets two yards on a quarterback sneak to keep the Eagles' drive alive.
</item 7>

<item 8>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
Viewers in Canada can stream every NFL game via DAZN.

Dolphins vs. Eagles start time

Date: Sunday, Oct. 22

Sunday, Oct. 22 Start time: 8:20 p.m. ET

Sunday night's game between the Dolphins and Eagles is set to kick off at 8:20 p.m. ET from Lincoln Financial Field in Philadelphia, Penn.

The Eagles haven't beaten the Dolphins since 2011, when Miami's loss resulted in the firing of coach Tony Sparano the following day. The Dolphins got the better
of Philadelphia in both 2015 and 2019, but none of those matchups carried the anticipation that this one does.
</item 8>

<item 9>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
Hurts gets two yards on a quarterback sneak to keep the Eagles' drive alive.

End of first quarter: Dolphins 3, Eagles 3

8:57 p.m. — DeVonta Smith has the Eagles into Dolphins territory after an eight-yard grab.

8:56 p.m. — Boston Scott with a really nice return to set up the Eagles at their own 39-yard line.

Dolphins 3, Eagles 3

8:52 p.m. FIELD GOAL — The Dolphins can't capitalize on the opportunity, so Jason Sanders ties the game with a field goal.

8:51 p.m. — A Tyreek Hill touchdown is erased by a holding penalty.

8:50 p.m. — The Eagles have lost yardage on three different plays in this quarter, which doesn't seem sustainable. It'll be 3rd & Goal from the 12.

8:48 p.m.
</item 9>

<item 10>
Dolphins vs. Eagles live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-22T23:28:04+00:00)
Check out all of the key moments you might have missed.

Dolphins vs. Eagles final score

1 2 3 4 F Dolphins 3 7 7 0 17 Eagles 3 14 7 7 31

Dolphins vs. Eagles results, highlights from 'Sunday Night Football'

(All times Eastern)

Final: Eagles 31, Dolphins 17

11:16 p.m. — And that'll do it. The Eagles finish strong and get back on track with an impressive win over the Dolphins.

11:12 p.m. — D'Andre Swift goes 22 yards for a first down, so the Eagles will simply kneel it the rest of the way.

11:09 p.m. — The Eagles take over on downs and will have the chance to just run out the clock on Miami.

11:07 p.m.
</item 10>
=== END ITEM id=77edc3a6772442f3 ===

=== ITEM id=c970e88ee155d343 ===
Gold sentences and facts:
Sentence 0: In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and effectively.
  Fact 0.0: Scientists point to a number of recent developments in the long term.
  Fact 0.1: The recent developments have raised hopes that dementia may be possible to tackle more directly.
  Fact 0.2: The recent developments have raised hopes that dementia may be possible to tackle more effectively.
Sentence 1: Health coach Susan Saunders found herself “parenting” her mother (who had dementia) while her own daughters were babies.
  Fact 1.0: Health coach Susan Saunders found herself parenting her mother.
  Fact 1.1: Susan Saunders's mother had dementia.
  Fact 1.2: Susan Saunders's own daughters were babies at the time.

Retrieved items:
<item 1>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Until now doctors have only been able to prescribe medicines that help patients manage their symptoms, so the arrival of the first drugs that treat the actual
cause of the condition has been welcomed – although experts have warned that their use should be treated with some caution.

After decades of research, these are the first drugs to improve patients’ lives directly. If nothing else, they suggest we’re on the right road to tackling
Alzheimer’s

“The new drugs slow down the development of Alzheimer’s by six months to a year and are useful only for those in the early stages of the condition, so they are
certainly not miracle medicines,” said David Thomas, head of policy at Alzheimer’s Research UK.

“However, after decades of research, they are the first to improve patients’ lives directly, and that is a justifiable cause for excitement. If nothing else,
they suggest we are probably on the right road to tackling Alzheimer’s.”

This point was backed by neurologist Cath Mummery, of the Dementia Research Centre at University College London.
</item 1>

<item 2>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It is a challenge, and a very important one.”
</item 2>

<item 3>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
However there are long waiting times – up to two years on average – for appointments at these centres.

In addition, diagnosing Alzheimer’s and other forms of dementia is usually based on pen-and-paper tests, followed by lumbar punctures and brain scans before a
final diagnosis.

About 65% of cases are confirmed this way. The remaining third of dementia cases are never diagnosed. Yet patients can only expect to receive treatments –
including the new drugs – if their condition is identified.

Eleanor Mackenzie-Smith’s father, Mike, has young-onset Alzheimer’s. His first symptoms were spotted 17 years ago, when Eleanor was 11. “However, it took over
10 years from the start of his symptoms and four separate tests between 2009 and 2017 to get my dad’s final diagnosis, when he was 65. It was distressing not
knowing what was happening. Too many families, like mine, have had to watch as dementia takes hold of our loved ones while we are left without a diagnosis and
the support and access to treatments.”
</item 3>

<item 4>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
People in Britain could benefit from a key medical breakthrough next year. They may be given access to the first drugs ever developed to slow the impact of
Alzheimer’s disease.

The first of these medicines – lecanemab – was recently approved in the US and Japan, where treatments using it have already been launched. A second drug,
donanemab, is expected to follow soon, and next year it is anticipated that the UK medical authorities will consider both of them for approval in Britain.

The prospect has raised hopes that, after years of effort, scientists may be closing in on ways to directly tackle the UK’s dementia crisis. About a million
people are living with the condition in this country, and this is expected to rise to about 1.7 million by 2040 – with potentially grim consequences. Last year
dementia took the lives of 66,000 people in England and Wales, and it is now the leading cause of death in Britain, with Alzheimer’s accounting for two-thirds
of cases.
</item 4>

<item 5>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and
effectively. One major challenge they have faced is the problem of getting drugs to pass through the blood-brain barrier, which controls the movement of ions
and molecules from our bodies to our brains. That makes it difficult to get drugs into the central nervous system to tackle problems such as amyloid plaques.

“However, researchers are developing active transport methods for getting drugs across the blood-brain barrier,” said Mummery. “For example, we are now
exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact.”

Such developments will take years to realise, scientists caution, and a great deal needs to be done in the short term to deal with dementia.

“It is clear we have taken a step in the right direction but there is much more to do before we can have successfully dealt with these processes that are
happening to our brains,” said Thomas.
</item 5>

<item 6>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
It took 10 years for Mike Mackenzie-Smith to get a diagnosis of young-onset Alzheimer’s.

Another grim illustration of the problem facing patients is provided by Graeme Armstrong. His wife, Trina, began having trouble recognising faces and reading
telephone numbers in 2006 and was told, after a CT scan three years later, that she had probably had a stroke – although her symptoms did not suggest such a
diagnosis. It took another three years before she was diagnosed with posterior cortical atrophy, a rare variant of Alzheimer’s disease that affects the way the
brain interprets information from the eyes.

“Had we had an accurate diagnosis four years earlier, Trina could have been placed on the correct medication, which may have been more effective and helped her
day to day,” Armstrong said.

One solution sought by doctors involves the creation of blood tests which could pinpoint the disease quickly and effectively. “These are under research, but we
are several years away from having them in widespread use,” said Thomas.
</item 6>

<item 7>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
Both drugs are to be considered for approval in the UK next year. The Medicines & Healthcare products Regulatory Agency (MHRA) will first decide if they are
safe and effective, then the National Institute for Health and Care Excellence (Nice) will rule on whether they offer value for money.

Both medicines are expensive – lecanemab costs about $25,000 (£19,700) a year – and are given by regular intravenous infusions. “That is a challenge from a
health-service point of view, because you need to find the space and time to put someone in an infusion suite to treat them,” added Thomas.

However, the main problem facing doctors is the difficulty involved in pinpointing dementia not just in its early stages but even in its later phases. Most
cases are first presented to GPs, who will then refer patients to memory clinics for a dementia test. However there are long waiting times – up to two years on
average – for appointments at these centres.
</item 7>

<item 8>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“Meanwhile, we have got to get the NHS to a state where it is much more focused and better organised in getting better and earlier dementia diagnoses. That will
be crucial in our battle against the condition.”

Mummery agreed: “By the time you have dementia, you will have had Alzheimer’s disease developing in your brain for at least 20 years. Dementia is the latest
stage; we need to pick up signals much earlier. So we need to be thinking about how to diagnose people at the very earliest stages of disease, when they may
only have very, very subtle symptoms.

We are now exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact Cath Mummery

“Our current service isn’t good for doing that, and we need to develop brain health clinics where we can pinpoint the disease much earlier, then help to build
resilience against dementia in a patient when it is only in a very early stage.”
</item 8>

<item 9>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It has been a very long, hard road, but finally we have something positive to look at. That is very welcome.”

Alzheimer’s disease is triggered by the buildup of a protein called amyloid in the brain, although symptoms may not appear for decades after this accretion has
started. Scientists have tried for more than 20 years to find ways to stop amyloid from forming these plaques, in the hope this would stop the progression of
the disease.

Lecanemab, which is produced by Japanese pharmaceutical company Eisai, and donanemab, produced by Eli Lilly of the US, are the first medicines to achieve this
aim – though they only slow but do not ultimately halt the disease’s progression.

Graeme Armstrong with wife Trina, who has a rare form of Alzheimer’s.

Both drugs are to be considered for approval in the UK next year.
</item 9>

<item 10>
Walks, tech and protein: how to parent your own parents (The Guardian, 2023-10-10T09:00:35+00:00)
“Anyone with older parents should be urging them out for a walk five days a week – and when they have a companion, the walk also doubles as a social occasion,”
he explained. “If they can walk for longer than 20 minutes, even better. Our study showed that the risk of depression fell in proportion to the time spent
walking.”

As well as a daily walk, Laird recommends vitamin D and B12 supplements – both of which appear to moderate the chance of depression in older people. “Depression
matters,” he added. “Not just because it reduces quality of life, but because in older people there seems to be a link between depression and dementia which
we’re still unpacking.”

Health coach Susan Saunders found herself “parenting” her mother (who had dementia) while her own daughters were babies. “I’d advise anyone with a parent in the
early stages of dementia to investigate all the available research trials. Start by joining Dementia Research, a nationwide register that matches volunteers to
research programmes.”
</item 10>
=== END ITEM id=c970e88ee155d343 ===

=== ITEM id=407cf1baf7a97761 ===
Gold sentences and facts:
Sentence 0: Announced on September 28 that the Fortnite maker is laying off 16% of its workforce, amounting to 870 people.
  Fact 0.0: The Fortnite maker announced on September 28 that it is laying off 16% of its workforce.
  Fact 0.1: The layoff amounts to 870 people.
Sentence 1: Because while Google spent most of its first day attempting to explain complicated ins and outs of business, Epic was able to paint a black-and-white picture of good and evil with itself as the clear underdog.
  Fact 1.0: Google spent most of its first day attempting to explain complicated ins and outs of business.
  Fact 1.1: Epic was able to paint a black-and-white picture of good and evil.
  Fact 1.2: In that picture Epic itself was the clear underdog.
  Fact 1.3: Epic was able to do this because Google spent most of its first day on those explanations.

Retrieved items:
<item 1>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
September 2023

IronNet

In a regulatory filing published September 29, the cybersecurity startup has ceased all business activities as it prepares for Chapter 7 bankruptcy and has laid
off its remaining staff.

Naughty Dog

Is reportedly cutting at least 25 contract developers, two sources familiar with the situation told Kotaku.

Epic Games

Announced on September 28 that the Fortnite maker is laying off 16% of its workforce, amounting to 870 people.

Talkdesk

An impacted employee confirmed on September 27 that at least 140 people, if not more, were affected by Talkdesk’s third round of layoffs in less than 14 months.

Byju’s

Announced on September 26, the company plans to cut 5,000 jobs due to restructuring efforts. The startup has eliminated more than 10,000 positions in the past
two years.

Roblox

Confirmed to TechCrunch on September 21, Roblox let go of 30 employees in its talent acquisition organization. No other teams are impacted.
</item 1>

<item 2>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
So there are absolutely no hard feelings, and we’re grateful that they joined the case because they did help in critical ways. Epic will continue to fight for
all developers, seeking remedies, and... I don’t know if Sundar is going to be calling me, but if he does, all of our discussions will be directed toward
solving the problem for everyone.

A lot has changed since you filed the original lawsuit over three years ago. For a bit there, it seemed like Epic had an interest in expanding outside of games
with acquisitions like Houseparty and Bandcamp. What changed? And did app store restrictions play into that at all?

Let’s see. No... well, not directly, anyway. We have a really broad strategy of building games and technology. We’ve had huge success in recent years with the
Unreal Engine gaining adoption among all kinds of industries even beyond games. We’ve also run into our own financial limitations. You know, we expanded the
company to nine times the size we were before Fortnite took off.
</item 2>

<item 3>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And when we tried to bundle Fortnite with other smartphone manufacturers like OnePlus and carriers of all sorts, they told us they couldn’t do a deal because
Google had done a secret deal with them.

It was really disconcerting to see the extent of bad faith efforts that were going on in a company of Google’s size. You’d think a trillion-dollar company would
develop to the point where they have pretty respectable processes and leadership structures that provide a check and balance against wrongdoing, but they were
rampantly destroying all their chats on these topics.

You’d see long conversation threads would start to get into a spicy antitrust issue, and suddenly somebody points out the history’s on and the chat goes silent.
They just turned it off to have the documents destroyed. It was great to see that all called out in detail.

Any particular evidence where you’re like, “Oh wow, I never saw that, and now that specific thing came out for the world to see”?

As an employee of Epic, I’ve not been able to see Google’s internal documents until the trial started.
</item 3>

<item 4>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?
</item 4>

<item 5>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.
</item 5>

<item 6>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This wasn’t the only layoff round of the year. Unity lowered its headcount three times in 2023, affecting about 900 jobs. In its quarterly financial results in
November, Unity reported a yearly revenue increase of 69 percent and it told investors, “We continued to manage costs well.”

Sony cut 100 jobs at Bungie, a company it bought for $3.6 billion in 2022. According to developers that are still there, Sony executives are attempting to use
this upheaval to wrest more control of the studio from Bungie founders and leaders.

Epic Games fired roughly 830 people this year, or 16 percent of its staff. This included significant job cuts at Mediatonic, the studio behind Fall Guys that
Epic purchased in 2021.

Mediatonic

“For a while now, we've been spending way more money than we earn,” CEO Tim Sweeney wrote about the layoffs.
</item 6>

<item 7>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Original story continues:

Llamacorn what now? What is happening?

Before we get to the Llamacorn, a little background. Epic is of course the studio behind Fortnite, the extraordinarily popular free-to-play game. Fortnite makes
money by selling in-game items with its virtual currency V-Bucks. Players often buy V-Bucks the same place they play Fortnite. And until August 13th, 2020, if
the player used an Android or iOS device and installed the game through an official app store, that purchase triggered an in-app payment fee to Google or Apple.

Critics call such fees the “Google tax” or the “Apple tax,” and Epic definitely wasn’t a fan.

When Epic decided to take action against these respective “taxes,” it made August 13th, 2020 a very busy day for Apple, Google, Epic, and us here at The Verge.
First, Epic announced it was bypassing Apple and Google’s app store fees.
</item 8>

<item 9>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 9>

<item 10>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What were your settlement talks with Google CEO Sundar Pichai like?

We can’t talk about content, but we met for an hour and had a professional discussion... in which we didn’t reach a settlement. We were rather far apart, let’s
say, because what Epic wants ultimately is free competition and fair competition for everybody, and the removal of the payments tie and removal of the
anticompetitive measures, which obviously leads to far better deals for consumers and developers.

What we don’t want is a special deal just for ourselves, and Google’s strategy so far as you’ve seen from all these deals has been to basically take out all
their potential competitors one at a time while they’re still weak and small, before they build up or unite into a formidable force. Epic’s not going to go
along with something like that and do a special deal just for ourselves.

You’ve said that a couple times, but the email that you accused Google of leaking suggested that Epic was interested in a special deal of some sort at that
time.
</item 10>
=== END ITEM id=407cf1baf7a97761 ===

=== ITEM id=296e37e2d46f89b1 ===
Gold sentences and facts:
Sentence 0: “She’s got the Eras Tour booked all through next year, and she’d rather die than cancel a show.
  Fact 0.0: The quoted woman (not named in this sentence) has the Eras Tour booked all through next year.
  Fact 0.1: The quoted woman would rather die than cancel a show.
Sentence 1: "It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something really major and traumatic happen to me and they don't know about it," she told NPR.
  Fact 1.0: The speaker (a woman not named in this sentence) said that it was almost like a very strange, subtle clue to the media.
  Fact 1.1: The speaker said the media do not know everything that happened in that relationship.
  Fact 1.2: The speaker said she can have something really major and traumatic happen to her without the media knowing about it.
  Fact 1.3: The speaker told NPR this.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.1: Swift has a new relationship with Kelce.

Retrieved items:
<item 1>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Since then, Swift has released four “Taylor’s Version” albums: Fearless (Taylor’s Version) in April 2021, Red (Taylor’s Version) in November 2021, Speak Now
(Taylor’s Version) in July 2023, and 1989 (Taylor’s Version) in October 2023.

“It’s not lost on me that the two great catalysts for this happening were two horrendous things that happened to me,” Swift said, when asked about the success
of her tour. “The first was getting cancelled within an inch of my life and sanity. The second was having my life’s work taken away from me by someone who hates
me.”

While Swift is arguably more famous now than ever before, the Grammy winner revealed that the constant attention and swarms of paparazzi won’t stop her from
living her life. There’s also constant rumours surrounding her personal life, specifically rumours about her relationship with Alwyn that her publicist, Tree
Paine, was forced to publicly deny.
</item 1>

<item 2>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 2>

<item 3>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
No-one, it turns out.

The song is thematically similar to Blank Space, commenting on the media's portrayal of Taylor's relationships: "But if I'm all dressed up / They might as well
be lookin' at us / And if they call me a slut / You know it might be worth it for once."

The best of the new tracks is Is It Over Now. An angrier take on 1989's philosophical break-up songs, it calls back to the Harry Styles snowmobile incident (see
also, Out Of The Woods) then, deliciously, references the "Sad Taylor Boat" meme - a photo of Taylor fleeing a vacation alone after the couple split up.

She goes on to call an ex a "lying traitor", accuses them of parading their new relationships in public, and noting that every new girlfriend looks like her.

"If she's got blue eyes, I will surmise that you'll probably date her."
</item 3>

<item 4>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
</item 4>

<item 5>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 5>

<item 6>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
The decision came after one of the singer’s fans, 23-year-old Ana Clara Benevides, died from a cardiac arrest during Swift’s first performance in the country.
Swift explained to Time that “part of my identity as a human” is putting on a show for her fans, no matter the circumstances. “I know I’m going on that stage
whether I’m sick, injured, heartbroken, uncomfortable, or stressed,” she said. “That’s part of my identity as a human being now. If someone buys a ticket to my
show, I’m going to play it unless we have some sort of force majeure.”

Perhaps it was her decision to re-record her music catalogue that catapulted Swift into such success. When music mogul Scooter Braun aquired Swift’s longtime
label, Big Machine Records, in 2019, he also gained the rights to the master recordings of Swift’s first six studio albums. In November 2020, Braun sold the
masters to an investment fund in a deal that was believed to be over $300 million.
</item 6>

<item 7>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter,” Swift told Time. “Yes, if I go out to dinner,
there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends.

“Life is short. Have adventures. Me locking myself away in my house for a lot of years - I’ll never get that time back,” she added. “I’m more trusting now than
I was six years ago.”
</item 7>

<item 8>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
In interviews, Taylor delighted in revealing this was based on a true story.

"It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something
really major and traumatic happen to me and they don't know about it," she told NPR.

In her Rolling Stone cover story, Taylor said the crash actually occurred on a snowmobile "with an ex who lost control and wrecked it so badly that she saw her
life flash before her eyes".

That prompted speculation the track was also about Styles: The couple had gone skiing with Justin Bieber and Selena Gomez in late 2012.

The lyrics also reference their identical aeroplane necklaces and a December break-up, which ties in with the "Sad Taylor on a boat" meme.

But the singer steadfastly refuses to name names: "The fact that I've never confirmed who those songs are about makes me feel like there is still one card I'm
holding," she told GQ.
</item 8>

<item 9>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
In an interview withWSJ Magazine, published on 20 November, he described how much he admires her skills as a musician. “I’ve never been a man of words. Being
around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

However, he also acknowledged the heavy public attention around his private life as a result of his relationship, noting he’s “never dated anyone with that kind
of aura around them”. But, according to Kelce, he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life. When she acts like
that I better not be the one acting all strange.”
</item 9>

<item 10>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has officially been named Time’s 2023 Person of the Year.

On Wednesday 6 December, the magazine announced Swift had earned the honour after an “editors’ assessment of the individual who most shaped the headlines over
the previous 12 months, for better or for worse”.

The recognition of Swift comes on the heels of a whirlwind year for the popstar, whose Eras Tour dominated ticket sales in 2023.
</item 10>
=== END ITEM id=296e37e2d46f89b1 ===

=== ITEM id=7140c556f2ba5780 ===
Gold sentences and facts:
Sentence 0: Hope has been rising on Wall Street recently that the US economy is slowing from its recent hot pace by just the right amount.
  Fact 0.0: Hope has been rising on Wall Street recently.
  Fact 0.1: The hope on Wall Street is that the US economy is slowing from its recent hot pace.
  Fact 0.2: The hope on Wall Street is that the US economy is slowing by just the right amount.
Sentence 1: A shutdown would drag on the US economy, raising the risk of a recession, though financial markets have held up relatively well through past shutdowns.
  Fact 1.0: A shutdown would drag on the US economy.
  Fact 1.1: A shutdown would raise the risk of a recession.
  Fact 1.2: Financial markets have held up relatively well through past shutdowns.

Retrieved items:
<item 1>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
“Considering the damage overnight in the US, Australian markets held on pretty well,” he said. “I thought it would be a bigger down day across the board.”
Glover said the market was volatile but quite range-bound, similar to previous periods of high inflation in the 1940s and 1970s. Elsewhere, Wall Street’s ugly
September got even worse on Tuesday, as a sharp drop for stocks brought them back to where they were in June. The S&P 500 tumbled 1.5 per cent for its fifth
loss in the last six days. The Dow Jones dropped 1.1 per cent, and the Nasdaq composite lost 1.6 per cent.

Loading September has brought a loss of 5.2 per cent so far for the S&P 500, putting it on track to be the worst month of the year by far, as the realisation
sets in that the Federal Reserve will indeed keep interest rates high for a long time.
</item 1>

<item 2>
ASX set for bright start despite Wall Street slip; $A drops (The Sydney Morning Herald, 2023-12-05T21:51:55+00:00)
Too much strength would give inflation more fuel, but too little would mean a recession.

With inflation down from its peak two summers ago, Wall Street is hopeful that the Federal Reserve may finally be done with its market-shaking hikes to interest
rates and could soon turn to cutting them. That could help the economy avoid a recession and give a boost to all kinds of investment prices. Investors have been
looking for a slowdown in the job market in particular. The hope is that it can cool more through employers cutting back on open positions than on employers
laying off workers. Tuesday’s report showed that employers advertised just 8.7 million jobs on the last day of October, down by 617,000 from a month earlier. In
the bond market, US Treasury yields continued to sag further from the heights they reached during late October. The yield on the 10-year Treasury fell to 4.18
per cent from 4.26 per cent late on Monday, offering more breathing space for stocks and other markets.
</item 2>

<item 3>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
That growing understanding has sent yields in the bond market to their highest levels in more than a decade, which in turn has undercut prices for stocks and
other investments. Treasury yields rose again on Tuesday following a mixed batch of reports on the economy. The yield on the 10-year Treasury edged up to 4.55
per cent from 4.54 per cent late on Monday and is near its highest level since 2007. It’s up sharply from about 3.5 per cent in May and from 0.5 per cent about
three years ago. One economic report on Tuesday showed confidence among consumers was weaker than economists expected. That’s concerning because strong spending
by US households has been a bulwark keeping the economy out of a long-predicted recession.

Besides high interest rates, a long list of other worries is also tugging at Wall Street. The most immediate is the threat of another US government shutdown as
Capitol Hill threatens a stalemate that could shut off federal services across the country.
</item 3>

<item 4>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
Wall Street is also absorbing the ouster of Kevin McCarthy as the speaker of the House of Representatives. The unprecedented move to remove a speaker from the
position likely doesn’t change much in the short term, with funding for the US government set until November 17.

“That said, a leadership vacuum in the House raises the odds of a government shutdown when the current funding extension expires,” according to economists at
Goldman Sachs. A shutdown would drag on the US economy, raising the risk of a recession, though financial markets have held up relatively well through past
shutdowns. On Wall Street, big tech stocks helped to support the market after leading it lower a day earlier. They tend to move more sharply with expectations
for rates because high-growth stocks are seen as some of the biggest victims of high yields. In markets abroad, stock indexes were mixed across much of Europe.
</item 4>

<item 5>
ASX set to open higher as Wall Street rebounds; $A rises (The Age, 2023-10-04T21:01:01+00:00)
That in turn could convince the Federal Reserve to take it easier on interest rates. After already raising its main interest rate to the highest level since
2001, the Fed has indicated it may keep its overnight rate higher next year than it had earlier expected. Treasury yields have correspondingly snapped higher as
traders accept a new normal for markets of high rates for longer. The Fed is paying particular attention to the job market because too much strength there could
drive wages for workers much higher, which it fears could keep inflation well above its target of 2 per cent.

Oil prices slumped overnight. Credit: AP Wednesday’s report from ADP suggested private employers added 89,000 jobs last month, a much sharper slowdown in hiring
than the 140,000 that economists expected. The report doesn’t have a perfect track record in predicting what the more comprehensive jobs report from the US
government will say. That will arrive on Friday.
</item 5>

<item 6>
ASX set to drop as Wall Street’s September slump deepens (The Sydney Morning Herald, 2023-09-26T19:11:30+00:00)
Loading Wall Street has dealt with such shutdowns in the past, and stocks have historically been turbulent in the run-up to them, according to Lori Calvasina,
strategist at RBC Capital Markets. After looking at the seven shutdowns that lasted 10 days or more since the 1970s, she found the S&P 500 dropped an average of
roughly 10 per cent in the three months heading into them. But stocks managed to hold up rather well during the shutdowns, falling an average of just 0.3 per
cent, before rebounding meaningfully afterward. Wall Street is also contending with higher oil prices, shaky economies around the world, a strike by US
autoworkers that could put more upward pressure on inflation and a resumption of US student-loan repayments that could dent spending by households.

On Wall Street, the vast majority of stocks fell under such pressures, including 90 per cent of those within the S&P 500.
</item 6>

<item 7>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
The National Australia Bank has forecast another rate hike to 4.35 per cent in November. Elsewhere, Wall Street closed out its worst month and quarter of the
year with more losses on Friday. The S&P 500 slipped 0.3 per cent and the majority of stocks within the index sank. The Dow Jones fell 0.5 per cent and the
Nasdaq composite edged 0.1 per cent higher. The Australian sharemarket is set to retreat, with futures pointing to a fall of 38 points, or 0.5 per cent, at the
open. Solid gains for stocks early on faded as pressure built from within the bond market. After easing earlier in the day on encouraging signals about
inflation, Treasury yields got back to rising as the day progressed. The yield on the 10-year Treasury yield returned to 4.58 per cent, where it was late
Thursday, after dipping to 4.52 per cent. It’s again near its highest level since 2007.
</item 7>

<item 8>
ASX set to open lower as Wall Street closes September with more losses (The Sydney Morning Herald, 2023-10-01T18:19:08+00:00)
Friday’s economic data shows that not only was inflation a touch cooler than expected in August, so was growth in spending by US consumers. That can be a
positive for inflation because it means not as many dollars are pouring into purchases. That in turn could give companies less encouragement to try to raise
prices further. But it may also dent what’s been a big driver keeping the US economy out of a recession.

“It came to a boil during a hot summer, and the temperature is really starting to come down,” said Brian Jacobsen, chief economist at Annex Wealth Management,
of spending growth by US consumers. “Higher energy prices, student loan debt repayments and real disposable incomes that have been on a declining trajectory
since June doesn’t bode well.” Oil prices have jumped to their highest level in more than a year, which is pressuring the economy by raising fuel costs for
everyone. A barrel of US crude sank 92¢ Friday to settle at $US90.79, but it’s still up sharply from $US70 in June.
</item 8>

<item 9>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
They also continued to decelerate from earlier months. A second economic report from the morning likewise offered both encouragement and caution for financial
markets. It said slightly fewer US workers than expected applied for unemployment benefits last week. On one hand, it indicates a job market with few lay-offs,
which means strength for the overall economy. But it could also be adding fuel to keep upward pressure on inflation.

After the reports, the 10-year Treasury yield rose to 4.7 per cent from 4.56 per cent late Wednesday. The two-year Treasury yield, which more closely tracks
expectations for the Fed, climbed to 5.06 per cent from 4.99 per cent. Yields are still down slightly for the week so far. After jumping last week to their
highest levels in more than a decade, yields regressed following speeches that investors saw as hints that the Fed may not raise its main overnight interest
rate further.
</item 9>

<item 10>
ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields (The Age, 2023-10-12T20:00:32+00:00)
Loading The big jump for the 10-year Treasury yield since the summer has pulled mortgage rates to their highest levels since the turn of the millennium, and
such moves could slow the economy without requiring more Fed hikes. Nothing in Thursday’s inflation report should sway the Fed one way or the other when it
comes to what it will do on November 1, when it announces its next move on interest rates, said Seema Shah, chief global strategist at Principal Asset
Management. She called the data “reassuringly uneventful”.

“The question around whether or not there will be one more interest rate hike is yet to be answered,” she said. Rising crude oil prices have put extra pressure
on inflation since the summer, and they were volatile again on Thursday. After jumping early in the day, a barrel of benchmark US crude slipped US58¢ to settle
at $US82.91. Brent crude, the international standard, rose US18¢ to $US86 a barrel.
</item 10>
=== END ITEM id=7140c556f2ba5780 ===

=== ITEM id=6823fff9775afded ===
Gold sentences and facts:
Sentence 0: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 0.0: Epic chose to focus on simply painting Google as the bad guy on day one.
  Fact 0.1: Epic's focus on painting Google as the bad guy happened on day one of the trial.
  Fact 0.2: The issue of painting Google as the bad guy will probably come up later in the trial.
Sentence 1: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 1.0: Epic will attempt to argue that Google restrains competition within two separate markets.
  Fact 1.1: One of the two markets is the distribution of apps to Android users.
  Fact 1.2: The other market is the market for payment processing solutions for content inside Android apps.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 3>

<item 4>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 4>

<item 5>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 5>

<item 6>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 6>

<item 7>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Didn’t other parties sue Google too?

They did! And then, they all settled. All 50 state attorneys general and a number of consumers have tentatively settled with Google over app store antitrust
claims, though it’s not clear what the deal might include.

Match Group, the company behind Tinder, Match.com, OkCupid and other dating apps, just reached a surprise last-minute settlement too, one that makes it sound
like Match effectively gave up.

Epic stands alone.

Could we get a settlement between Google and Epic as well?

Epic seems to have no interest in settling as long as the “Google tax” exists, and the company’s repeatedly said it’s not interested in cash. So far, the
company’s been willing to push these cases all the way through multiple appeals, cost be damned.

It’s not clear what Google could proactively offer that would satisfy Epic, since CEO Tim Sweeney has also publicly suggested that Google would simply
circumvent any settlement it offers.
</item 8>

<item 9>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 9>

<item 10>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 10>
=== END ITEM id=6823fff9775afded ===
