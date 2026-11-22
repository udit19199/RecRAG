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

=== ITEM id=2d27d1910fcbf96c ===
Gold sentences and facts:
Sentence 0: You're probably starting him if Kenneth Walker III remains out with his oblique injury, but consider other options if you have the depth.
  Fact 0.0: Kenneth Walker III has an oblique injury.
  Fact 0.1: Kenneth Walker III may remain out with his oblique injury.
  Fact 0.2: You are probably starting him if Kenneth Walker III remains out with his oblique injury.
  Fact 0.3: Consider other options if you have the depth.
Sentence 1: With Kenneth Walker III's oblique injury comes opportunity for the rookie, who already looked strong in the month leading up to Seattle's narrow Week 11 loss to the Rams.
  Fact 1.0: Kenneth Walker III has an oblique injury.
  Fact 1.1: Kenneth Walker III's oblique injury creates opportunity for the rookie.
  Fact 1.2: The rookie already looked strong in the month leading up to Seattle's narrow Week 11 loss to the Rams.
  Fact 1.3: Seattle lost narrowly to the Rams in Week 11.
Sentence 2: Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
  Fact 2.0: Kenneth Walker III plays for the Seahawks.
  Fact 2.1: Kenneth Walker III took four carries against the Rams.
  Fact 2.2: Kenneth Walker III left the game with an oblique injury.
  Fact 2.3: Kenneth Walker III left the game after taking four carries against the Rams.

Retrieved items:
<item 1>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 1>

<item 2>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.
</item 2>

<item 3>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
The former All-Pro, who battled a lingering hamstring injury earlier this year, received attention on the sidelines after limping off, then left the game for
further evaluation in the locker room. He was ruled questionable to return. NFL Media reported on Monday that Kupp suffered a low ankle sprain and will be
evaluated going forward.

Darius Slayton (Giants) had a big impact against the Commanders, hauling in a deep score from Tommy DeVito, before exiting with an arm injury. He was ruled out
for the remainder of the contest halfway through the third quarter.

EDGE

Joey Bosa (Chargers) appeared to injure his right leg or foot while held in a block during a first-quarter pass-rushing attempt against the Packers. He limped
to the sidelines and was carted to the locker room, then ruled questionable to return.
</item 3>

<item 4>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
Half of the six targets were uncatchable. In his previous three with Deshaun Watson he had at least seven targets in each, and at least 16 PPR points in the
past two.

SINCE THEN: Watson hasn't practiced with the team. P.J. Walker was named the starter despite being on the team's practice squad and took reps this week in
preparation for their game against the Niners.

WALKER: Was cut by the Bears after the preseason before linking up with Cleveland. Has five career passing touchdowns in 15 games (seven starts). He's had
multiple passing scores in one NFL game. Only twice has a receiver had 100 yards from Walker (both by D.J. Moore, once in 2020, once in 2022). Two other times
has a WR had at least 70 yards (Curtis Samuel in 2020, Terrace Marshall in 2022).
</item 4>

<item 5>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.
</item 5>

<item 6>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Kansas City's front-seven has been tough against speedsters and more vulnerable to power backs, and the 5-11, 190-pound Cook 100 percent falls into the former
category. Temper your expectations.

We feel the same way about Zach Charbonnet (@ 49ers). It's great that the Week 13 injury he sustained on Thursday night was just a bruise, but it's not great
that he now faces a 49ers defense that already stifled him on Thanksgiving. You're probably starting him if Kenneth Walker III remains out with his oblique
injury, but consider other options if you have the depth. The floor and ceiling remain low against this uber-confident Niners D.

Tyler Allgeier (vs. Buccaneers) continues to see his role diminish for the NFC South-leading Falcons, with dum-dum coach Arthur Smith finally realizing that
Bijan Robinson gives the ATL the best chance at continued success.
</item 6>

<item 7>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 7>

<item 8>
Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:07:00+00:00)
The injury highlights the importance of handcuffs, as Zack Moss has put up multiple top-10 performances with and without Taylor this season.

Seattle's Kenneth Walker III (oblique) missed Week 12 and 13, and we're not confident in his ability to suit up against the hard-hitting 49ers coming off a big
win in Philly. Backup Zach Charbonnet exited early because of a knee injury against the Cowboys, putting his status in doubt, too. If both are out, DeeJay
Dallas would start and lead the backfield in touches.

Green Bay's Aaron Jones has a sprained MCL, and sources have revealed that the Packers continue to consider him week-to-week. He only logged four carries before
sustaining the injury in Week 11, then he sat for the Packers' Thanksgiving Day tilt with the Lions and their Sunday Night Football win over Kansas City. Why
rush him back if Green Bay is already winning games?
</item 8>

<item 9>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
They weren't awful against Dallas last week on a per-touch basis, but neither finished with more than 12 touches. Even worse, no Carolina back has enjoyed a
60-yard rushing game since Oct. 15. Bad offensive line, negative game scripts, awful coaching — it's all bad, man!

What are the biggest RB injuries in Week 12?

Second-year Seahawks back Kenneth Walker III suffered an oblique injury in Week 11 that the ever-hip Pete Carroll referred to as "legit." KW3 has a better
chance of missing multiple weeks than he does at suiting up for Seattle's Thanksgiving Day game against Fred Warner and the 49ers. If Zach Charbonnet somehow
finds himself on your waiver wire, an early Merry Christmas to you if you have the FAAB or a high waiver priority.

The hits just keep on coming for the Packers' Aaron Jones, who suffered a knee injury but seems to have at least caught a break in the sense that he didn't tear
his ACL.
</item 9>

<item 10>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
Tampa Bay has a pretty good run defense, so Allgeier can't be trusted outside of a desperate flex play with his nine average touches over the past two weeks.

What are the biggest RB injuries in Week 14?

Derrick Henry (head/neck), Rhamondre Stevenson (ankle), Kenneth Walker III (oblique), Aaron Jones (knee), D'Onta Foreman (ankle), and D'Andre Swift
(undisclosed) are all "questionable," at best, to suit up this week. We know for sure Stevenson will be out, but we'll have to monitor the injury reports
throughout the week for the rest.

Jonathan Taylor (thumb) underwent surgery last week and will miss the next few weeks, opening the door for Zack Moss to resume his role as Indy's super-sub lead
back.

We'll be updating these RB PPR rankings all week, so check back for the latest player movement.
</item 10>
=== END ITEM id=2d27d1910fcbf96c ===

=== ITEM id=7ab4e0edb7f77f7c ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made big improvements with its new iteration.
  Fact 0.2: The minor physical upgrades are part of the new iteration alongside the big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: Valve looks at a store that could be both.
  Fact 1.1: Valve decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units ready to ship on November 16th.

Retrieved items:
<item 1>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 1>

<item 2>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 2>

<item 3>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Valve Valve Steam Deck OLED SCORE 89 Engadget 89 Critics - Not yet scored N/A Users - Not yet scored N/A Pros Improved battery life

Bright and smooth visuals

Better thermals

Feels less top-heavy

Long charging cable Cons Still ridiculously large

Seriously it’s so big

Valve Valve Steam Deck OLED This isn't the Steam Deck 2 some are waiting for, it's just a welcome improvement on an already good device. $549 at Steam

OLED screen

The HDR OLED display looks fantastic. It features a wider P3 color gamut, pure blacks and a peak brightness of 1,000 nits, compared with the original max of 400
nits. In action, this means the screen’s bezels meld with the black of any game, creating a more polished frame for retro aspect ratios and providing room for
higher-contrast colors. The screen on the new model doesn’t support variable refresh rates (VRR), but it now tops out at 90Hz, while the original could only hit
60Hz.
</item 3>

<item 4>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 4>

<item 5>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 5>

<item 6>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 6>

<item 7>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 7>

<item 8>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is the highlight of the device, while things like faster charging, improved antennas and smoother haptics are welcome bonuses. More
importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the granddaddy of handheld PCs and Valve
has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2, which may be the final version of the
device — unless Valve finally learns how to count to three.
</item 8>

<item 9>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The original Steam Deck had a 7-inch 800p LCD running at 60Hz. The new model squeezes a 7.4-inch OLED screen into the same space, significantly shrinking the
bezels around the display. The new panel runs at the same resolution, but with a refresh rate of up to 90Hz. It’s also HDR, covering 110 percent of the P3 color
space with a peak brightness of 1,000 nits — the old model maxed out at around 400 nits. Like all OLED screens, it has pure blacks and near-instantaneous
response times. As before, anti-glare etched glass is available, but only on the premium model.

The Steam Deck OLED has a die-shrunk version of the same custom APU found in the original Steam Deck. The new APU is produced on a 6nm process, rather than the
older 7nm process.
</item 9>

<item 10>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly —
in just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to
build hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies
like Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).

So, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the facetuned,
photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.
</item 10>
=== END ITEM id=7ab4e0edb7f77f7c ===

=== ITEM id=7bbb2bbc2982a8bf ===
Gold sentences and facts:
Sentence 0: Altman’s exit “is indeed shocking as he has been the face of” generative AI technology, said Gartner analyst Arun Chandrasekaran.
  Fact 0.0: Altman's exit is indeed shocking
  Fact 0.1: Altman has been the face of generative AI technology
  Fact 0.2: Arun Chandrasekaran, a Gartner analyst, said this
Sentence 1: Altman has been telling investors that he’s planning to launch a new venture, according to The Information.
  Fact 1.0: Altman has been telling investors that he is planning to launch a new venture
  Fact 1.1: The Information is the source of this report
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory is based on the board's language
  Fact 2.1: The prevailing theory is based on the way giant tech companies work
  Fact 2.2: The prevailing theory is floating around right now

Retrieved items:
<item 1>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
That’s according to Bloomberg, which in a brief — citing sources close to the matter — said that discussions are happening between Quora CEO Adam D’Angelo, one
current member of the OpenAI board, and Altman — and possibly other board members as well.

Board tensions boil over

The New York Times reports that, before his ousting, Sam Altman made a move to push out board member Helen Toner because he thought a paper she had co-written
was overly critical of OpenAI. That, among other issues, led to OpenAI’s current predicament. Speaking of, The Times indicates that negotiations to hire Altman
back continue — but that one major sticking point remaining is “guardrails” meant to improve Altman’s communication with the board.

—

November 20

Altman joins Microsoft

Sam Altman, Greg Brockman and colleagues announce that they’ll join Microsoft to lead a new AI research team.
</item 1>

<item 2>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Altman has a long history steering OpenAI. After co-founding the company alongside Peter Thiel, LinkedIn cofounder Reid Hoffman and others, Altman, who formerly
headed startup accelerator Y Combinator and also holds a seat on the board at Worldcoin, the ambitious cryptocurrency project, initially served as a co-chair of
OpenAI alongside Elon Musk. Musk left in 2018 to avoid a conflict of interest with Tesla.

Altman over the past several months played an active role in attempting to shape regulators’ responses to AI, appearing at U.S. congressional hearings and
meeting personally with world leaders including President Joe Biden, German Chancellor Olaf Scholz and French President Emmanuel Macron, among others, on an
international policy tour.

More important than being one of the most brilliant and impactful people our industry has ever had Sam is one of the most generous and caring people I know.
I've never met someone who has supported and lifted up more people around them than him. Couldn't be a prouder brother.
</item 2>

<item 3>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
In a dramatic turn of events late Friday, ex-Y Combinator president Sam Altman was fired as CEO of AI startup OpenAI, the company behind viral AI hits like
ChatGPT, GPT-4 and DALL-E 3, by OpenAI’s board of directors. Then, the company’s longtime president and co-founder, Greg Brockman, resigned — as did three
senior OpenAI researchers. And the fallout continues.

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

It’s a fast-moving situation that we’re still trying to get to the bottom of. No doubt more will become clear as time goes on. To make it easier to follow all
that’s happened in the meantime, though, we’ve put together a timeline; we’ll do our best to keep it current.
</item 3>

<item 4>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Confirmation Sam Altman will not return as OpenAI’s CEO

A number of investors and OpenAI employees tried to bring back Altman after his sudden firing by the company’s board, but following a weekend of negotiations,
it was confirmed that Altman would not return to OpenAI and new leadership would take hold. What this means for ChatGPT’s future, and for the OpenAI Dev Day
announcements, remains to be seen.

Sam Altman ousted as OpenAI’s CEO

Sam Altman has been fired from OpenAI. He will leave the company’s board and step down as CEO, with OpenAI’s chief technology officer Mira Murati stepping in as
interim CEO. In a blog post from OpenAI, the company writes that the board “no longer has confidence in [Altman’s] ability to continue leading OpenAI.”

In a statement on X, Altman said working at OpenAI “was transformative” for him and “hopefully the world.”
</item 4>

<item 5>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
The Information, speaking to three sources formerly with the company, reports that they no longer expect the sale — led by Thrive Capital — to happen, or, if it
does, to come with a lesser valuation because of the recent turn of events.

Altman planning new venture

Altman has been telling investors that he’s planning to launch a new venture, according to The Information. Brockman is expected to join the effort — whatever
form it takes. (Possibly an AI chip startup.)

i love you all. today was a weird experience in many ways. but one unexpected one is that it has been sorta like reading your own eulogy while you’re still
alive. the outpouring of love is awesome. one takeaway: go tell your friends how great you think they are.
</item 5>

<item 6>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
We are collaborating to figure out the details. Thank you so much for your patience through this. — OpenAI (@OpenAI) November 22, 2023

Altman also posted about the new deal, giving some insight into the roller coaster that we’ve all been riding since his firing was revealed Friday. In his
words, his decision to join Microsoft on Sunday “was the best path for me and the team.” Since then, the new board’s composition and Microsoft’s support appear
to have been enough to bring him back to the AI company he co-founded.

Altman and board in talks

OpenAI’s board of directors is reportedly in talks with Sam Altman, ex-Y Combinator president and an OpenAI co-founder, to return to OpenAI as CEO as soon as
this week.
</item 6>

<item 7>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Later Monday, that number climbed to over 650.

Altman and Brockman considering return

As reported by The Verge, Altman’s move to Microsoft isn’t a done deal — and both Altman and Brockman are still open to returning to OpenAI. That is, if the
remaining board members who initially fired him step aside.

OpenAI board considers merger

OpenAI’s board of directors approached Dario Amodei, the co-founder and CEO of rival large-language model developer Anthropic, about a potential merger of the
two companies, The Information reports. The approach was part of an effort by OpenAI to persuade Amodei to replace Altman as CEO — but Amodei quickly turned
down the CEO offer.

—

November 19

Altman to meet at OpenAI HQ

According to The Information, Altman is expected to meet at OpenAI’s San Francisco headquarters as executives at OpenAI push to have him reinstated as CEO.
</item 7>

<item 8>
Sam Altman ousted as OpenAI’s CEO (TechCrunch, 2023-11-17T20:41:13+00:00)
Sam Altman has been fired from OpenAI, Inc., the 501(c)(3) nonprofit that acts as the governing body for OpenAI, the AI startup behind ChatGPT, DALL-E 3, GPT-4
and other highly capable generative AI systems. He’ll both leave the company’s board of directors and step down as CEO.

In a post on OpenAI’s official blog, the company writes that Altman’s departure follows a “deliberative review process by the board” that concluded that Altman
“wasn’t consistently candid in his communications” with other board members, “hindering its ability to exercise its responsibilities.”

Tip TechCrunch

Do you work at OpenAI and know more about Sam Altman’s departure?

Do you work at OpenAI and know more about Sam Altman’s departure? Get in touch with TechCrunch

“The board no longer has confidence in [Altman’s] ability to continue leading OpenAI,” the blog post reads.
</item 8>

<item 9>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Timeline of Sam Altman’s firing from OpenAI

November 29

Microsoft gains a board observer

Microsoft will gain representation on the new initial board of directors in the form of a non-voting observer, OpenAI announced. It wasn’t immediately clear who
this observer might be — only that they won’t have an official vote in board business.

November 21

Sam Altman, OpenAI reach agreement on return as CEO and ‘initial’ new board

In a sudden late announcement, OpenAI revealed that it and Altman “have reached an agreement in principle” for him to return as the company’s CEO. In addition
to Altman’s return, its new “initial” board will include former Salesforce chief executive Bret Taylor, former US Secretary of the Treasury Larry Summers and
Quora founder Adam D’Angelo.

We have reached an agreement in principle for Sam Altman to return to OpenAI as CEO with a new initial board of Bret Taylor (Chair), Larry Summers, and Adam
D'Angelo.
</item 9>

<item 10>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
After sending a Google Meet link, Brockman was told that he was being removed from the board as chairman “but was vital to the company and would retain his
role” as president, and that Altman had been fired.

Altman’s firing publicly announced

OpenAI published a post on its blog announcing the executive shake-up. The company’s management team was aware shortly after.

i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved working with such talented people.
will have more to say about what’s next later. 🫡 — Sam Altman (@sama) November 17, 2023

All-hands meeting

OpenAI held an all-hands meeting Friday afternoon during which Sutskever defended Altman’s ouster. He dismissed suggestions that pushing Altman out amounted to
a “hostile takeover,” and claimed that it was necessary to protect OpenAI’s mission of “making AI beneficial to humanity.”
</item 10>
=== END ITEM id=7bbb2bbc2982a8bf ===

=== ITEM id=369671c33748ada1 ===
Gold sentences and facts:
Sentence 0: India were playing against Australia in the second semi-final of the 2015 World Cup in Sydney on March 26.
  Fact 0.0: India played against Australia in the second semi-final of the 2015 World Cup.
  Fact 0.1: The India versus Australia match was held in Sydney.
  Fact 0.2: The India versus Australia match was played on March 26.
Sentence 1: It's crunch time for England in the World Cups as they face South Africa in the semi-final of the Rugby World Cup and in the ODI Cricket World Cup.
  Fact 1.0: It is crunch time for England in the World Cups.
  Fact 1.1: England faces South Africa in the semi-final of the Rugby World Cup.
  Fact 1.2: England faces South Africa in the semi-final of the ODI Cricket World Cup.

Retrieved items:
<item 1>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
England vs. South Africa betting odds, prediction

The last time these two sides met was a tumultuous period to say the least, particularly for English rugby. South Africa's dominant 27-13 victory at Twickenham
proved to be the last game in charge for Eddie Jones, the coach who had taken England to the 2019 World Cup final. In his place, Steve Borthwick was hired, and
he's just one game away from repeating that feat.

England's form at this tournament has been perfect — at face value. Argentina, Japan, Chile, Samoa and Fiji have all provided highlights — Los Pumas are also in
the semi-finals — and England have beaten all of them. However, only Japan and Chile were defeated with bonus points, and the narrow wins over both Samoa and
Fiji in their last two games showed that England are certainly gettable.
</item 1>

<item 2>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
India moved to finals and beat Sri Lanka to win their second World Cup title.



2015 World Cup, against Australia

India were playing against Australia in the second semi-final of the 2015 World Cup in Sydney on March 26. Australia outplayed India in almost all departments
of the game.

Australian Man-of-the-Match Steve Smith (105 runs in 93 balls) and Aaron Finch (81 runs in 116 balls) were the guiding forces in taking Australia to 328 runs
for 7 wickets in 50 overs.

In reply, India needed a few big innings, but apart from skipper MS Dhoni (65 runs in 65 balls), Shikhar Dhawan (45 runs in 34 balls), and Ajinkya Rahane (44
runs in 68 balls), the batting order collapsed miserably as India lost the match by 95 runs.



2019 World Cup, against New Zealand

The cricket World Cup returned to England for the fifth time.
</item 2>

<item 3>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
Also Read: IND vs NZ FREE Live Streaming: When and How to watch India vs New Zealand Cricket World Cup 2023 Semi-Final Match Live on Web, TV, mobile apps online

Also Read: AUS vs SA FREE Live Streaming: When and How to watch Australia vs South Africa Cricket World Cup 2023 Semi-Final Match Live on Web, TV, mobile apps
online

1983 Prudential Cup, against England

Having lost in the first round of the 1975 and 1979 World Cups, India were underdogs in the third World Cup, which was being held in England for the third
consecutive time.

However, in the World Cup, India wronged many of their critics, reaching semi-finals and playing against mighty hosts England in Manchester on June 22.

Skipper Kapil Dev struck three wickets while Roger Binny and Man-of-the-Match Mohinder Amarnath took two each as India restricted England to 213 all out in 60
overs.
</item 3>

<item 4>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
TV channel Streaming USA — Peacock Canada TSN TSN+ UK ITV ITV X Australia Channel 9 Stan Sport, 9Now New Zealand Sky GO Sky GO India — Fan Code Hong Kong BeIN
Sports BeIN Sports Malaysia BeIN Sports BeIN Sports Singapore BeIN Sports BeIN Sports

MORE: Every single team to have lifted the Rugby World Cup trophy

England vs South Africa lineups, team news

"The start is crucial, the end is vital..."



- Steve Borthwick explains changing 3️⃣ of his England starters for Saturday's Rugby World Cup semi-final, from Liam Heagney 👨‍💻 in Paris. #RWC2023 #ENGvRSA
#EnglandRugby
</item 4>

<item 5>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
They moved into the last-four stage on the next two occasions when the World Cup was held in India, in 1987 and 1996.

India were in semi-finals in 2003 in South Africa.

Eight years later, they were again in last four at the 2011 World Cup as hosts, and repeated their feat in 2015, 2019, and 2023.

It means that in their last six World Cups, India have been in the semi-finals on five occasions.

Also Read: ICC Cricket World Cup 2023 semi-final schedule: India to play New Zealand in Mumbai; South Africa to face Australia in Kolkata; know venue, squads

In this write-up, ZeeBiz will take you through the memory lane of India's semifinal journey in the biggest cricket extravaganza in the world.
</item 5>

<item 6>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
In reply, Mohinder Amarnath played a useful innings of 46, while Yashpal Sharma scored a fine 61.

Sandeep Patil played a fiery innings of 51 not out in just 32 balls as India stormed into semifinals for the first time, beating England by six wickets with 5.2
overs to spare.

Also Read: ICC World Cup 2023: List of centuries; Most tons in World Cup so far

1987 Reliance Cup, against England

Four years later, the World Cup was being played outside England for the first time, and it was India which was hosting it.

In a co-incidence, India was against England in the semi-finals once again, at the Wankhede Stadium, Mumbai, on November 5.

This time, England's Man-of-the-Match skipper Graham Gooch smashed 115 runs in 136 balls to lay a solid foundation for his team's fighting total of 254 for 6
wickets.
</item 6>

<item 7>
ICC World Cup 2023: India at Cricket World Cup semi-finals so far (Zee Business, 2023-11-11T19:12:00+00:00)
2023 World Cup, against New Zealand

To be played
</item 7>

<item 8>
Liverpool vs Everton begins epic Saturday with World Cup semi-final as well as Chelsea vs Arsenal – and it could end with a bang (TalkSport,
2023-10-20T21:34:28+00:00)
talkSPORT will have updates from all the action in Paris as England take on the Springboks from 8:00pm

9 Owen Farrell will be looking to inspire England to victory against the Springboks Credit: Getty

Over in India, England's Cricket World Cup hopes hang by a thread after their latest shock defeat to Afghanistan.

They will be boosted by the news that Ben Stokes is fit for the match after he picked up a hip injury five days before the tournament.

But victory for England could scarcely be more crucial.

Having also lost to New Zealand in their opener, their chances of finishing in the top four and progressing to the semi-finals are looking dubious.

But South Africa, despite winning their first two games, suffered a shock defeat to the Netherlands in their most recent match - a sign for England that the
Proteas can be beaten.
</item 8>

<item 9>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The 2019 Rugby World Cup final between England and South Africa was one of many firsts: the game itself capped off a sensational tournament played outside of
the sport's traditional strongholds for the first time; it was the first final in which South Africa scored a try, and by full-time, the 32-12 score in the
Springboks' favour sealed a record-equalling third title for the Rainbow Nation.

Fast forward four years, and South Africa once again prepare to take on England in an all-or-nothing match to decide which nation gets a shot at World Cup glory
— but this time around, the pair meet in the semi-finals of the 2023 tournament, rather than the decider.

That won't make an ounce of difference for either camp, though, as these two long-time rivals prepare for a battle that could re-define the definition of a
'physical encounter.'
</item 9>

<item 10>
Where to watch England vs South Africa: Live stream, TV channel, lineups, odds for 2023 Rugby World Cup semifinal (Sporting News, 2023-10-20T22:00:05+00:00)
The last time Cheslin Kolbe played England at a Rugby World Cup... 🇿🇦 🏆#RWC2023 | #ENGvRSA pic.twitter.com/Riw2uaSaKa — Rugby World Cup (@rugbyworldcup) October
19, 2023

Both sides' journeys have seen twists, turns, and bumps in the road, but it's led them both to within 80 minutes of a Rugby World Cup final. England will no
doubt be looking to echo the spirit of 2019, when they beat the All Blacks 19-7 to reach the final, but in South Africa, they face a side who have taken their
game to new heights just to reach the semis.

MORE: 2023 Rugby World Cup final: Details, kick-off time and every team's route

What time does England vs South Africa kick off?
</item 10>
=== END ITEM id=369671c33748ada1 ===

=== ITEM id=21b44909becdff23 ===
Gold sentences and facts:
Sentence 0: Unfortunately, she did say that she had conversations with investors as part of their due diligence — and, of course, Alameda was taking on losses from FTX to keep FTX’s balance sheet pristine.
  Fact 0.0: The witness said she had conversations with investors as part of their due diligence.
  Fact 0.1: Alameda was taking on losses from FTX to keep FTX's balance sheet pristine.
Sentence 1: It was not, however, possible for those accounts to avoid liquidation, as Wang testified Alameda could do — or to have an overall negative balance.
  Fact 1.0: It was not possible for those accounts to avoid liquidation.
  Fact 1.1: Wang testified that Alameda could avoid liquidation for those accounts.
  Fact 1.2: It was not possible for those accounts to have an overall negative balance.
Sentence 2: It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy and sell cryptocurrencies.
  Fact 2.0: The argument described Alameda as a 'very innovative, successful company'.
  Fact 2.1: The argument's speaker claimed the business practices between Alameda and FTX were reasonable.
  Fact 2.2: The argument's speaker claimed Alameda acted legally as an FTX customer.
  Fact 2.3: The argument's speaker claimed Alameda acted legally as an FTX payment processor.
  Fact 2.4: The argument's speaker claimed Alameda acted legally as an FTX market maker.
  Fact 2.5: A market maker is a financial entity that acts as a trading partner for customers looking to buy and sell cryptocurrencies.

Retrieved items:
<item 1>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Wang said he trusted Bankman-Fried’s judgment and didn’t pursue the issue further. Later, though, Alameda’s balance was more negative than FTT and the trading
revenue, and Wang had more conversations with Bankman-Fried about it. Wang said he knew at the time that Alameda was using FTX’s customers’ funds and that he
knew it was wrong.

We heard yesterday about the line of credit Alameda had on FTX, a staggering $65 billion. Today we saw it in the database. The line of credit didn’t start that
big — Wang testified that Bankman-Fried had asked him to increase it a few times because Alameda kept running into its limits. First it was “a few hundred
million,” then a billion, then even that wasn’t enough. It wasn’t clear how Wang arrived at $65 billion specifically, but he said he’d discussed the number with
Bankman-Fried. Other customers didn’t have the same privileges.
</item 1>

<item 2>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It was turned on in July. No other customer trading on FTX had “allow_negative” privileges — only Alameda. Wang’s testimony suggests that the fraud at FTX
started very early since the same day that allow_negative was switched on, Bankman-Fried tweeted that Alameda’s account “was like everyone else’s.”

The general sense I got, from the testimony of Bankman-Fried’s former friends, was that things just kept escalating. For instance, Alameda Research first had a
negative balance in late 2019, Wang testified. Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money
from the account as long as the amount was less than FTX’s total trading revenue. A little later, Wang discovered that Alameda was negative by $50 million more
than FTX’s revenue. Bankman-Fried told him it was fine, and it just didn’t account for the FTT token that the two of them had launched. Wang trusted him.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
We have some inkling of what the government will be arguing thanks to the filings prosecutors are making in this case. They argue that Bankman-Fried lied about
consumer protection and that Bankman-Fried’s statements that FTX was “avoiding or managing conflicts of interest” and that “as a general principle FTX
segregates customer assets from its own assets across our platforms” were lies, in a superseding indictment filed August 14th.

The indictment also says that there were special features in FTX’s code that “permitted Alameda to spend and withdraw unlimited amounts of money from FTX,”
which were created at Bankman-Fried’s direction. That effectively exempted Alameda from the kinds of risk management other customers faced. At the same time,
the government alleges, “Bankman-Fried publicly and repeatedly asserted that Alameda did not have privileged access to FTX.”
</item 4>

<item 5>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Then a bigger one. Then an effectively limitless one.

Ellison testified to something similar, choking back tears: “When I started working at Alameda, I don’t think I would have believed you if you told me I would
be sending false balance sheets to our lenders or taking customer money, but over time, it was something I became more comfortable with.”

Ellison’s tenure as CEO of Alameda Research was an exercise in optics, she testified. Alameda Research was, by then, FTX’s sin-eater. When FTX took a loss in
the hundreds of million dollars because a trader had exploited its system, Wang moved that loss onto Alameda’s balance sheet. FTX’s “insurance fund,” meant to
keep traders from getting stuck with other traders’ losses, was totally made up — Alameda simply picked up the tab.

Even that wasn’t good enough.
</item 5>

<item 6>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Those privileges ultimately resulted in Alameda owing FTX $8 billion worth of customer deposits. "We gave special privileges on FTX that gave unlimited
withdrawals on the platform to Alameda," Wang said. Alameda was allowed to withdraw and transfer those funds and had a $65 billion line of credit. "When
customers deposited USD, it went to Alameda," he said. "It existed in the computer code. Alameda could have negative balances and unlimited withdrawals." That
"bug" in the code was written by Nishad Singh, who was FTX's director of engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang
also told the court about a $1 million personal loan he received and a $200 million to $300 million loan in his name from Alameda that was never deposited into
his account, but rather was used to make investments into other companies on behalf of FTX.
</item 6>

<item 7>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
In late 2019, Alameda had a negative balance on FTX. Because they sat in an open-plan office, Wang heard a trader ask Bankman-Fried if it was okay to keep
withdrawing money from that account. Bankman-Fried said it was okay as long as it was lower than FTX’s total trading revenue. He was the CEO of both companies
at the time.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses.

The amount that Alameda was allowed to go negative began to creep up. Around 2019 or 2020, Wang checked the database and discovered that Alameda was negative by
about $200 million, which was more than the $150 million FTX made in revenue. That had to mean Alameda was taking customer money. It surprised him and he says
he discussed it with Bankman-Fried, who told him that he just needed to take into account the value of FTT. Wang said he trusted Bankman-Fried’s judgment and
didn’t pursue the issue further.
</item 7>

<item 8>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
Other customers didn’t have the same privileges.

Alameda also allegedly kept FTX from looking bad by assuming some of its losses. In 2021, FTX was facing a “several hundred million dollar” loss from an exploit
in MobileCoin, Wang said. (This seems to have happened in April 2021, though Wang didn’t give a month.) Bankman-Fried directed him to have Alameda take it on
instead, because FTX’s balance sheets were visible to investors and Alameda’s weren’t.

Yesterday, Paradigm Capital’s Matt Huang showed us the balance sheets he received from Bankman-Fried. They didn’t include the loss, which should have been
included in the second quarter numbers he’d received — they showed $63 million in trading expenses and $63 million in “other expenses” in that quarter, and an
estimated profit of $732 million.

“FTX was not fine and assets were not fine.”

By 2022, Alameda’s negative balance on FTX was a cause for concern for Caroline Ellison, Singh, and Wang.
</item 8>

<item 9>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
</item 9>

<item 10>
Today the FTX jury suffered through a code review (The Verge, 2023-10-06T22:21:46+00:00)
FTX lied about how much money was in the backstop fund

In the process of liquidating, FTX would try to sell the collateral on the open market, but if it couldn’t finish, then backstop liquidity providers would step
in. These were market makers, including Alameda, which could be compensated for the losses they take from a “backstop fund,” Wang said.

But FTX lied about how much money was in the backstop fund, Wang said. In court we saw the code that generated the fake number published on the website: it took
the daily trading volume on FTX, multiplied it by a random number, divided it by a billion, and added it to the existing number displayed on the site. It had
nothing to do with the actual amount of money in the insurance fund.

And when there wasn’t enough money in the fund, money was moved there from Alameda’s accounts in order to pay the insurance out, Wang said.

In late 2019, Alameda had a negative balance on FTX.
</item 10>
=== END ITEM id=21b44909becdff23 ===

=== ITEM id=8e3cc2e06e2f0b96 ===
Gold sentences and facts:
Sentence 0: And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.0: Valve has made a gaggle of minor physical upgrades with its new iteration.
  Fact 0.1: Valve has made these minor physical upgrades alongside the big improvements.
  Fact 0.2: Valve's new iteration includes big improvements.
Sentence 1: We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.
  Fact 1.0: The writer looks at Valve and sees a store that could be both.
  Fact 1.1: Valve has decided to focus exclusively on games.
Sentence 2: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 2.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 2.1: Valve has units of the Steam Deck OLED ready to ship on that same day.

Retrieved items:
<item 1>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.
</item 1>

<item 2>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.
</item 2>

<item 3>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 3>

<item 4>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 4>

<item 5>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I think the philosophy change would do both of those companies much good.

If you get your way in terms of being able to freely put your own store on Android, do you believe that would only be a game store or would it also be an app
store? We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.

So the Epic Games Store isn’t a games store, right? It’s the store operated by Epic Games. So we have a lot of non-games there already. We have the Brave web
browser, we have a number of software creation tools including Unreal Engine, and there’s more coming, including some other awesome creation tools and
productivity tools. We’ll host any app anybody wants of any sort.

I think the gaming market is something we’re uniquely close to, and so I think we would likely be able to forge closer partnerships and opportunities in gaming,
but we’ll be open to everybody on Android as we are on PC.
</item 5>

<item 6>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 6>

<item 7>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 7>

<item 8>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.
</item 8>

<item 9>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 9>

<item 10>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
</item 10>
=== END ITEM id=8e3cc2e06e2f0b96 ===
