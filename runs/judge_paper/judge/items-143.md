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

This file has 5 items.

=== ITEM id=945e8ef862ecbc51 ===
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
“Meanwhile, we have got to get the NHS to a state where it is much more focused and better organised in getting better and earlier dementia diagnoses. That will
be crucial in our battle against the condition.”

Mummery agreed: “By the time you have dementia, you will have had Alzheimer’s disease developing in your brain for at least 20 years. Dementia is the latest
stage; we need to pick up signals much earlier. So we need to be thinking about how to diagnose people at the very earliest stages of disease, when they may
only have very, very subtle symptoms.

We are now exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact Cath Mummery

“Our current service isn’t good for doing that, and we need to develop brain health clinics where we can pinpoint the disease much earlier, then help to build
resilience against dementia in a patient when it is only in a very early stage.”
</item 1>

<item 2>
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
</item 2>

<item 3>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
“It is a challenge, and a very important one.”
</item 3>

<item 4>
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
</item 4>

<item 5>
New Alzheimer’s drugs bring hope of slowing disease for UK patients (The Guardian, 2023-12-17T10:00:23+00:00)
People in Britain could benefit from a key medical breakthrough next year. They may be given access to the first drugs ever developed to slow the impact of
Alzheimer’s disease.

The first of these medicines – lecanemab – was recently approved in the US and Japan, where treatments using it have already been launched. A second drug,
donanemab, is expected to follow soon, and next year it is anticipated that the UK medical authorities will consider both of them for approval in Britain.

The prospect has raised hopes that, after years of effort, scientists may be closing in on ways to directly tackle the UK’s dementia crisis. About a million
people are living with the condition in this country, and this is expected to rise to about 1.7 million by 2040 – with potentially grim consequences. Last year
dementia took the lives of 66,000 people in England and Wales, and it is now the leading cause of death in Britain, with Alzheimer’s accounting for two-thirds
of cases.
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
In the long term, scientists also point to a number of recent developments which have raised hopes that it may be possible to tackle dementia more directly and
effectively. One major challenge they have faced is the problem of getting drugs to pass through the blood-brain barrier, which controls the movement of ions
and molecules from our bodies to our brains. That makes it difficult to get drugs into the central nervous system to tackle problems such as amyloid plaques.

“However, researchers are developing active transport methods for getting drugs across the blood-brain barrier,” said Mummery. “For example, we are now
exploring ways of helping drugs get into the brain much more efficiently, and that could have a major impact.”

Such developments will take years to realise, scientists caution, and a great deal needs to be done in the short term to deal with dementia.

“It is clear we have taken a step in the right direction but there is much more to do before we can have successfully dealt with these processes that are
happening to our brains,” said Thomas.
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
</item 10>
=== END ITEM id=945e8ef862ecbc51 ===

=== ITEM id=9b3ebdac4642653d ===
Gold sentences and facts:
Sentence 0: In the above scenario, supply disruption will lead to higher inflation and central bankers will have to adjust their policies and hold rates higher for longer than anticipated.
  Fact 0.0: In the above scenario, supply disruption will lead to higher inflation.
  Fact 0.1: In the above scenario, central bankers will have to adjust their policies.
  Fact 0.2: In the above scenario, central bankers will have to hold rates higher for longer than anticipated.
Sentence 1: Back then, just like today, home prices had boomed for years before Fed officials were ultimately forced to hike interest rates aggressively in an attempt to fight inflation.
  Fact 1.0: Back then, home prices had boomed for years before Fed officials were ultimately forced to hike interest rates aggressively.
  Fact 1.1: Fed officials were ultimately forced to hike interest rates aggressively in an attempt to fight inflation.
  Fact 1.2: The home price boom back then was like the home price boom today.

Retrieved items:
<item 1>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
After the most aggressive monetary-tightening campaign in four decades, academics and economic practitioners are running autopsies on what could have prevented
the cost-of-living crisis and how to ensure the same mistakes won’t be repeated.

Markets have scrambled to price in high-for-longer interest rates, with a new war in the Middle East adding yet more risk to an already uncertain outlook
confronting central bankers as they gather for their penultimate meetings of a tumultuous year.

The policy navel-gazing is centering around three debates. How much flexibility central banks can allow in reaching their inflation targets, the effectiveness
of asset purchases in the policy mix, and the merits of monetary and fiscal coordination.

Bloomberg surveyed economists from around the world to gather views on those three debates. Their verdict: Central banks won’t break their economies in a rush
to hit inflation targets, QE will be used more sparingly in the future, and fiscal policy risks countering the work of monetary authorities.
</item 1>

<item 2>
ASX to rise after interest rate hopes boost US shares (The Sydney Morning Herald, 2023-11-05T20:56:22+00:00)
The Fed put such pressure on the economy intentionally, hoping to starve inflation of its fuel. It wants the job market to cool, particularly pay raises going
to workers. The Fed fears too-strong pay gains could create a vicious cycle that keeps inflation high.

Analysts said Friday’s jobs report offered encouraging signals for the Fed, with average hourly earnings rising less in October from September than expected,
though it doesn’t mean the job is done. Treasury yields in the bond market tumbled immediately after the jobs report, releasing more of the pressure that had
built up on Wall Street. The yield on the 10-year Treasury eased to 4.52 per cent from 4.67 per cent late Thursday and from more than five per cent last week,
when it hit its highest level since 2007. In the US, traders are moving up expectations for when the first cut to interest rates by the Fed could happen,
potentially by the summer, according to data from CME Group. Such cuts can act like steroids for financial markets.
</item 2>

<item 3>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
The ECB faces an extra legal burden on bond holdings that comes with operating in a currency union of 20 countries. Concerns around illegally financing
governments and debt mutualization have already landed the central bank in court several times.

Mixing Policies

Low interest rates and large-scale QE programs allowed treasuries to borrow on the cheap to finance stimulus campaigns, protecting labor markets, businesses and
consumers from collapse. But the spending blowout throughout and since the pandemic — part critical emergency funding, part political need to show an
all-hands-on-deck approach in crisis — contributed to the latest outbreak in inflation.

While the same kind of pulling in the same direction is needed to restrain demand, many governments are concerned that if they tighten policy too hard, voters
will kick them out and replace them with populists or extremists. That’s reviving questions about whether central banks can deliver price stability all on their
own.
</item 3>

<item 4>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
QE effectively swaps long-term borrowing costs for short-term ones. What’s been a lucrative deal for taxpayers when official interest rates were low has now
turned into a disastrous trade.

The clearest depiction of the problem is in the UK, where the BOE secured taxpayer indemnity for any losses on QE. Over the next decade, it estimates, its
purchases will cost the government over £200 billion ($243 billion).

And policymakers have little experience in unwinding their balance sheets, where small mistakes can trigger big market turbulence.

The Fed experienced some of that when it tried to shrink bond holdings between 2017 and 2019. More recent efforts to reduce portfolios have progressed rather
smoothly, partially because central banks have amassed so much debt over the years that they’re far away from any thresholds that would trigger a squeeze.

But the fact that they’re treating quantitative tightening as a technical adjustment rather than a part of their efforts to conquer inflation raises questions
about the future use of a tool that’s only trusted to work one way.
</item 4>

<item 5>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
That’s reviving questions about whether central banks can deliver price stability all on their own.

“If we were designing optimal policy arrangements from scratch, monetary and fiscal policy would both have a role in managing the economic cycle and inflation,
and that there would be close coordination,” Philip Lowe said in his last speech as Australian central bank governor in September.

Economists surveyed by Bloomberg predict fiscal policy will somewhat counteract the Fed’s efforts to rein in inflation in the US.

“It’s true that there are circumstances where working hand in hand and supporting each other has proved helpful,” ECB President Christine Lagarde told a panel
discussion in June at the institution’s annual economic forum.

Fed Chair Jerome Powell, who sat to her right, signaled he wasn’t ready to rely on that kind of cooperation. “Our assignment is to deliver price stability kind
of regardless of the stance of fiscal policy.”

Central bankers warn that any failure to scale back fiscal spending risks coming at the cost of yet higher interest rates.
</item 5>

<item 6>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
She recommends they be ready to react preemptively, even when inflation hasn’t yet spun out of control.

They may be called into action soon on that front, should an escalation in the conflict in the Middle East hit oil deliveries.

When the next big global slowdown comes, though, flexibility may be needed the other way. Europe’s eight-year experiment with negative rates ended with mixed
reviews last summer as to whether it was all worth it.

The Bank for International Settlements argues that there’s room for greater tolerance for moderate shortfalls even if they’re persistent, because “low-inflation
regimes, in contrast to high-inflation ones, have self-stabilizing properties.”

Rethinking Quantitative Easing

With a more flexible approach to those 2% targets, monetary policy after the 2008 financial crisis would have looked very different in many parts of the world.
</item 6>

<item 7>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
Olivier Blanchard, a former IMF chief economist, has long argued in favor of raising the inflation target, and former European Central Bank Vice President Vitor
Constancio has also embraced the idea. But it’s a controversial view and only possible from a position of credibility, which means central banks would likely
have to get inflation back to 2% first.

“It would be a mistake of the first order to think you can change a goal you have set if you can’t achieve it,” according to Bundesbank President Joachim Nagel.

Global trends suggest inflation will be stronger than in the past, with former Bank of England Governor Mark Carney among those saying rates won’t return to
pre-pandemic lows.

One lesson Gita Gopinath, the IMF’s No. 2 official, draws from the latest inflation episode is that policymakers mustn’t assume that looking through supply
shocks — as text books suggest — is the optimal response.
</item 7>

<item 8>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
They also want elected officials to put in place policies that help deliver sustainable growth.

“A change in mindset needs to happen,” said Agustin Carstens, the former governor of the Bank of Mexico who’s now the general manager of the BIS. “Growth needs
to depend less on fiscal and monetary policy, it should depend more on structural policies.”
</item 8>

<item 9>
Economists, central bankers are rethinking the quantitative easing strategy that saved the world but created bubbles and distortions (Fortune,
2023-10-22T15:26:11+00:00)
What Bloomberg Economics Says… “A long period of galloping price gains, and fears that the last yards back to target could be most painful for workers, have
reignited the debate about whether central banks should aim for a higher rate of inflation. That’s a conversation worth having. But for monetary policymakers,
the imperative of retaining credibility means the right time for it is after inflation is back at target, not before.” — Tom Orlik, chief economist

Rethinking Targets

So long as people believe prices will get back toward 2%, central bankers have some leeway in deciding how aggressive they need to be in pursuing that goal.

Economists covering 16 of the world’s most important central banks say policymakers will allow more time to bring inflation back to target if it means less
damage to their economies. The Bloomberg special survey also shows that a sizable minority sees them going even further, accepting price pressures that are
either slightly too strong or too weak — as long as expectations remain anchored.
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
=== END ITEM id=9b3ebdac4642653d ===

=== ITEM id=e3b2423b614151fc ===
Gold sentences and facts:
Sentence 0: "And if the Braves let this be the story of their series, that's on them.
  Fact 0.0: A quoted speaker says that if the Braves let this be the story of their series, that is on the Braves.
Sentence 1: But while Nocturne could have easily settled and coasted on classic Castlevania vibes, the new show steps its game up considerably by using the franchise’s iconic lore to challenge your understanding of what it truly means to fight for freedom, equality, and fraternity.
  Fact 1.0: Nocturne could have easily settled and coasted on classic Castlevania vibes.
  Fact 1.1: The new Nocturne show steps its game up considerably.
  Fact 1.2: The new Nocturne show uses the Castlevania franchise's iconic lore.
  Fact 1.3: The new Nocturne show uses the franchise's iconic lore to challenge the viewer's understanding of what it truly means to fight for freedom, equality, and fraternity.

Retrieved items:
<item 1>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
Nocturne reworks a number of Castlevania’s classic supporting characters like Maria — who fights using reimagined versions of the Four Sacred Beasts — and
empathic singer Edouard (Sydney James Harcourt) to brilliant effect. But out of all the show’s old but new faces, ferrokinetic Vodoun practitioner Annette
(Thuso Mbedu) stands out because of how strongly the show introduces her as a woman whose Haitian identity is what makes her able to understand the situation
over in France with an exacting kind of clarity.

It’s through Annette and the show’s exploration of Afro Caribbean syncretism that Nocturne expands the boundaries of Castlevania’s magical mythology, which is
touched on heavily throughout the season’s eight episodes. Small nods to the video games and gnarly beheadings abound, particularly in the Powerhouse
Animation-produced show’s bigger hyperkinetic brawls between its heroes and hordes of horrific creatures.
</item 1>

<item 2>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
It had been reported Monday night when the Braves beat the Phillies in Game 2 of the NLDS, ending on a critical base-running play involving Bryce Harper, that
Atlanta shortstop Orlando Arcia taunted about the Phillies' star in the clubhouse after the game.

That report wound up generating plenty of controversy.

Fox Sports' Jake Mintz, one of the two brothers that runs the popular Cespedes Family BBQ account, reported that Arcia had been celebrating after Atlanta's 5-4
win by saying "Atta boy Harper" in the Braves' clubhouse, jeering at the fact that Harper was doubled off on the basepaths after an outstanding catch by Michael
Harris in center field and an impressive relay to get Harper at first to end the game.

The comments wound up coming back to bite the Braves two days later, with Harper homering twice — staring down Arcia after each — as he powered the Phillies to
the 10-2 Game 3 win.
</item 2>

<item 3>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
But in all the bloodshed, Nocturne still finds the time to make its characters (and by extension, you) think about what it means to slaughter technically
innocent people for a genuinely noble cause. That beat isn’t entirely original, and Nocturne’s mindful about leaning on it too hard as it begins to weave
Annette and Edouard’s stories together with Richter and Maria’s in France. Still, it creates a resonance between the show’s action and the concepts it’s
tackling that makes the season as a whole work beautifully.

Image: Netflix

Occasionally, there are moments where the fluid dreaminess of Nocturne’s overall art direction butts up against the show’s application of 3D animation to make
certain pivotal moments pop. It can be jarring when traditionally animated characters who were just popping in and out of view using tricks of the eye suddenly
turn to the camera with deeper faces accented by flat shadows that weren’t there a second before.
</item 3>

<item 4>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
As strong and different as the season starts, by its last two episodes, Nocturne does start to feel a bit like it’s rehashing moments from the first show — if
not in exact form, then in terms of how neatly it brings its plotlines to a close. None of that’s enough to keep the show from hitting all the right notes,
however, and by the time you hit the finale, there’s a solid chance you’ll already be hoping for more.
</item 4>

<item 5>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
But despite all of Julia’s skill with her family’s deadly heirloom, Castlevania: Nocturne’s world is one teeming with vampires even centuries after Dracula has
seemingly been destroyed, and the series opens at a devastating moment in Richter’s past that leaves him orphaned and unable to access his magic.

Though Nocturne takes place in the same continuity as the previous Castlevania show, it also uses major pieces of lore from Konami’s Castlevania: Rondo of Blood
and Symphony of the Night games to tell a riveting story about the inherent hypocrisies baked into the historic record of colonial nations embracing democratic
revolution.

Here, the French Revolution isn’t just a transformative period of social unrest between moneyed elites and impoverished peasants fed up with their subjugation.
</item 5>

<item 6>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
She said that reporters abstaining from writing on his comments would be akin to "protecting players from themselves."

She also went on to say that what Arcia said wasn't a slur or hateful, and added the only reason anyone was concerned about it was because "they know it would
motivate Harper to beat [the Braves]. Which implies the Braves do not feel they can handle a motivated Bryce Harper and are willing to suggest that is why they
lost the other day."

"I didn't say something sooner because I thought this was overblown," Janes continued. "And frankly, I thought it wouldn't help. But I am genuinely disgusted by
the way some media members are acting like a reporter was in the wrong for this. He wasn't.

"And if the Braves let this be the story of their series, that's on them. Win and it goes away. Period."
</item 6>

<item 7>
Bryce Harper 'Atta Boy' controversy, explained: Why Orlando Arcia's comments led to backlash, Alanna Rizzo rant (Sporting News, 2023-10-13T21:58:11+00:00)
The only difference was that Janes did not name the player.

The reporting of Arcia's comments proved to be divisive across the baseball community. Among the most prominent to call out Mintz was Rizzo. The "High Heat"
co-host said she was irritated by "bloggers or podcasters" going into the clubhouse and "ruining it for the rest of us" when "some jackoff comes in at the end
of the season that gets a credential, God only knows why." She called the clubhouse "a sacred place" and defended the players who were upset by the reporting of
Arcia's comments, before she called Mintz out by name.

MORE: Every time Nick Castellanos has interrupted a somber moment with a home run

"First of all, this guy, Jake Mintz, that's not even a reporter," Rizzo said. "That's taking away from true reporters and true journalists.
</item 7>

<item 8>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
Part of / The best entertainment of 2023

Netflix’s first Castlevania animated series excelled by balancing its quiet, beautifully performed moments of human drama with explosive and artful action set
pieces dazzling enough to evoke the feeling of playing Konami’s classic side-scroller for the first time. Despite the first show’s strengths, its regular issues
with pacing and effectively cultivating characters from season to season made the idea of Netflix keeping the vampiric daddy issues ball rolling with
Castlevania: Nocturne — a new spinoff series from executive producer Clive Bradley and co-directors Sam Deats and Adam Deats — a dubious one.

Nocturne both looks and feels a lot like its predecessor in terms of its focus on a tortured scion waging war against machiavellian, bloodsucking ghouls who
fancy themselves undead gods who walk among mortal livestock.
</item 8>

<item 9>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
But while Nocturne could have easily settled and coasted on classic Castlevania vibes, the new show steps its game up considerably by using the franchise’s
iconic lore to challenge your understanding of what it truly means to fight for freedom, equality, and fraternity.

Set in the late 18th century during the thick of the French Revolution, Castlevania: Nocturne picks up hundreds of years after the events of Netflix’s first
Castlevania series as demon slayer Richter Belmont (Edward Bluemel) makes his way to France in search of vengeance and vampires. Like his distant holy
whip-wielding ancestor Trevor, Richter’s innate talents for magic make him perfectly suited for getting into the family business, and being raised by his
formidable warrior mother Julia (Sophie Skelton) instills in him a profound sense of pride in the work they do.

Image: Netflix

Together, Richter’s raw potential and Julia’s tutelage make him uniquely powerful even from a young age.
</item 9>

<item 10>
Castlevania: Nocturne is a revolutionary epic that does the franchise’s best games justice (The Verge, 2023-09-28T20:06:12+00:00)
It’s a war between undead ghouls like Drolta Tzuentes (Elarica Johnson) manipulating mortals from the shadows, selfish humans like a power-hungry Abbot
(​​Richard Dormer), and agitators like summoner Maria (Pixie) who want to dismantle the entire oppressive system of power that makes France a formidable global
presence.

Image: Netflix

Narratively, Nocturne’s focus on another brooding Belmont and its depiction of late-18th-century France as a country overrun by literally vampiric aristocrats
drunk on the promise of a new messiah make the new series feel true to the essence of Castlevania. What makes Nocturne’s deep dive into this alternate history a
worthwhile watch, though, is the way it presents France not just as a crumbling monarchy in the throes of a heroic social uprising but also as a colonial nation
whose pivot to democracy at home can’t be divorced from its history of committing unfathomable atrocities abroad.
</item 10>
=== END ITEM id=e3b2423b614151fc ===

=== ITEM id=5938db4447d10f32 ===
Gold sentences and facts:
Sentence 0: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 0.0: Epic will also suggest that Google hid items from discovery by allowing executives and employees to have 'off the record' chats.
  Fact 0.1: Google will counter this, saying it has provided the court with thousands of chat logs.
  Fact 0.2: Google says it has provided the court with millions of documents.
Sentence 1: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 1.0: Google did not release the full model, Gemini Ultra.
  Fact 1.1: Google released only a 'lite' version called Gemini Pro.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that this happens through anticompetitive means.

Retrieved items:
<item 1>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
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
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 3>

<item 4>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 4>

<item 5>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.
</item 5>

<item 6>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 6>

<item 7>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.
</item 7>

<item 8>
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
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 10>
=== END ITEM id=5938db4447d10f32 ===

=== ITEM id=54b5ed0b354ea83f ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: The deals Google makes include deals with Apple, Samsung, Mozilla, and many others.
  Fact 0.2: The Justice Department is focused on these deals to ensure Google is the default search engine on practically every platform.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: The last roundup reported how Google spent $26.3 billion in 2021 making Google the default search engine across platforms.
  Fact 1.1: The last roundup reported how Google tried to have Chrome preinstalled on iPhones.

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 1>

<item 2>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
“We try to get people up and running as fast as possible.”

Cue acknowledged that the ISA didn’t allow Apple to offer users a choice of search engines during setup but also said he wouldn’t want to do that anyway. “We
try to get people up and running as fast as possible,” he said. “Setup is just critical stuff.” Showing people a bunch of search engines they’ve never heard of
would just be a bad user experience, he argued; even Cue couldn’t remember the names of some of the alternatives to Google. “We make Google be the default
search engine,” he said, “because we’ve always thought it was the best. We pick the best one and let users easily change it.” (“Easily” is a persistent point of
contention in this trial — DuckDuckGo’s CEO, who testified last week, claimed it takes “too many steps” to switch.)

As for the privacy pop-ups? This is where Bellshaw began to press on how exactly Apple decided Google had the best product.
</item 2>

<item 3>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Eddy Cue, in a dark suit, peered down at the monitor in front of him. The screens in the Washington, DC, courtroom had briefly malfunctioned and left witnesses
with only binders, but now the tech was up and running — showing an image of three iPhones, each demonstrating a part of the phone’s setup process. Cue squinted
down at the screen.

“The resolution on this is terrible,” he said. “You should get a Mac.” That got some laughs in an otherwise staid and quiet courtroom. Judge Amit Mehta,
presiding over the case, leaned into his microphone and responded, “If Apple would like to make a donation…” That got even bigger laughs. Then everybody got
back down to business.

Cue was on the stand as a witness in US v. Google, the landmark antitrust trial over Google’s search business.
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 4>

<item 5>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.
</item 5>

<item 6>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Cue is one of the highest-profile witnesses in the case so far, in part because the deal between Google and Apple — which makes Google the default search engine
on all Apple devices and pays Apple billions of dollars a year — is central to the US Department of Justice’s case against Google.

Cue had two messages: Apple believes in protecting its users’ privacy, and it also believes in Google. Whether those two statements can be simultaneously true
became the question of the day.

Apple is in court because of something called the Information Services Agreement, or ISA: a deal that makes Google’s search engine the default on Apple’s
products. The ISA has been in place since 2002, but Cue was responsible for negotiating its current iteration with Google CEO Sundar Pichai in 2016. In
testimony today, the Justice Department grilled Cue about the specifics of the deal.

When the two sides renegotiated, Cue said on the stand, Apple wanted a higher percentage of the revenue Google made from Apple users it directed toward the
search engine.
</item 6>

<item 7>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 7>

<item 8>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 8>

<item 9>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 9>

<item 10>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 10>
=== END ITEM id=54b5ed0b354ea83f ===
