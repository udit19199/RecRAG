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

This file has 3 items.

=== ITEM id=35962659466f7d09 ===
Gold sentences and facts:
Sentence 0: Defense analysts agree that much of the outlook for Ukraine is dependent on the outcome of the U.S. vote.
  Fact 0.0: Defense analysts agree that much of the outlook for Ukraine depends on the outcome of the U.S. vote.
Sentence 1: Any illusion of the success of globalization, or of the twenty-first century representing a break from the brutal twentieth century, was stripped away with Russia’s most recent invasion of Ukraine.
  Fact 1.0: Any illusion that globalization had succeeded was stripped away with Russia's most recent invasion of Ukraine.
  Fact 1.1: Any illusion that the twenty-first century represented a break from the brutal twentieth century was stripped away with Russia's most recent invasion of Ukraine.
  Fact 1.2: The invasion of Ukraine was Russia's most recent invasion.

Retrieved items:
<item 1>
With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024 (Cnbc | World Business News Leader, 2023-12-25T23:36:00+00:00)
Aid and politics

Another year of war in Europe has undoubtedly drained Western military resources and the political appetite to maintain massive amounts of military aid for
Ukraine. Ongoing funding for Ukraine is far from secure in 2024 given the fact that the U.S. presidential election could herald a seismic change in the attitude
toward and support for Kyiv. Specifically, all eyes are on former U.S. President Donald Trump, the Republican front-runner, who cultivated close relations with
his Russian counterpart Vladimir Putin during his presidency. There are concerns that, given Trump's previous good relations with Moscow and "America First"
policy, aid for Ukraine could be shelved rapidly. Defense analysts agree that much of the outlook for Ukraine is dependent on the outcome of the U.S. vote. "I
think it's important to understand the extent to which Ukraine is reliant on the U.S. right now, because it's quite significantly more reliant on the U.S.

Graph facts:
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> Russia
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> Russia
- Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. -MENTIONS-> Russia
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Russia
- Dissecting impact of Israel-Hamas conflict on markets -MENTIONS-> Russia

Entities: Russia, military aid, Ukraine, Vladimir Putin, Europe, CNBC, Kyiv, U.S. presidential election, With hopes of victory fading, Ukraine's fight against
Russia could get even harder in 2024, war in Europe, Moscow, America First policy, Defense analysts, U.S., Donald Trump
</item 1>

<item 2>
With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024 (Cnbc | World Business News Leader, 2023-12-25T23:36:00+00:00)
Away from the battlefield, military experts said that the trajectory the Russia-Ukraine war takes in 2024 will mostly be dictated thousands of miles away in the
U.S., Ukraine's largest military supporter, and whether aid declines in the run-up — and following — the U.S. presidential election. "War is an uncertain
endeavor," retired Army Lt. Gen. Stephen Twitty, former deputy commander of the U.S. European Command, told CNBC. "Russia can win the war, or the Ukrainians can
win the war. And, as you're seeing things now, if you really think about it, what has been achieved this year? Very little has been achieved by Russia, and you
can say the same thing for the Ukrainians," he said.

Ukrainian servicemen take part in a military training exercise not far from front line in the Donetsk region on June 8, 2023.

Graph facts:
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> Russia
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> Russia
- Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. -MENTIONS-> Russia
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Russia
- Dissecting impact of Israel-Hamas conflict on markets -MENTIONS-> Russia

Entities: Russia, Ukraine, CNBC, Stephen Twitty, U.S. European Command, Russia-Ukraine war, U.S. presidential election, Donetsk region, military training
exercise, With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024, U.S.
</item 2>

<item 3>
With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024 (Cnbc | World Business News Leader, 2023-12-25T23:36:00+00:00)
Anatolii Stepanov | Afp | Getty Images

"We're in this situation now where if there's not a clear winner, there's going to be a stalemate, and there's going to be, perhaps, a future frozen conflict.
What can tilt the balance, in my view, is if the Ukrainians are not resupplied and they're not re-funded and they don't get the equipment and people that they
need. Then this war could tilt to the Russians," Twitty noted.

Expectations not met

A year ago, Ukraine's international military support was solid with NATO pledging to support Kyiv for "as long as it takes" as it defended itself against
Russia's invasion launched in February 2022. Over the summer, however, the challenge facing Ukraine's forces was glaringly obvious as they struggled to break
through heavily fortified Russian positions and lines of defenses along a swathe of the 600-mile-long front line across the southern and east of the country.

Graph facts:
- Manchester United face Galatasaray with high hopes but bad memories -MENTIONS-> AFP
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> AFP
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> AFP
- Inside the Man United takeover and it’s anything but smooth sailing for Sir Jim Ratcliffe as he seeks board vote -MENTIONS-> AFP
- Leaky defence ‘isn’t good enough’ and ‘scars’ grow from throwing away leads – It’s all going wrong for Manchester United in Europe -MENTIONS-> AFP

Entities: AFP, Getty Images, Russia, Ukraine, CNBC, Kyiv, Anatolii Stepanov, Twitty, NATO, Russia's invasion, With hopes of victory fading, Ukraine's fight
against Russia could get even harder in 2024, war
</item 3>

<item 4>
With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024 (Cnbc | World Business News Leader, 2023-12-25T23:36:00+00:00)
right now, because it's quite significantly more reliant on the U.S. than it is on the EU," Sam Cranny-Evans, defense analyst at the Royal United Services
Institute defense think tank, told CNBC. "If the U.S. election goes in a way that is not in Ukraine's favor, coupled with the fact that the EU is not really
stepping up to the plate — it's ammunition production is so far off what it should have been by now to give Ukraine a hope of surviving and a hope of victory —
it's not a very cheery prediction for 2024."

Good chemistry: President Donald Trump and Russian President Vladimir Putin shake hands during a joint press conference after their summit on July 16, 2018 in
Helsinki, Finland. Chris McGrath | Getty Images News | Getty Images

Rumblings of discontent over continuing Ukraine aid have been heard in some Republican quarters for months now, as well as in Eastern Europe. Former U.S.

Graph facts:
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> EU
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> EU
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> EU
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> EU
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> EU

Entities: EU, Russia, Ukraine, United States, Vladimir Putin, CNBC, Republicans, With hopes of victory fading, Ukraine's fight against Russia could get even
harder in 2024, Sam Cranny-Evans, Royal United Services Institute, ammunition production, U.S. election, summit, Helsinki, Finland
</item 4>

<item 5>
With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024 (Cnbc | World Business News Leader, 2023-12-25T23:36:00+00:00)
Those things have to happen. And I think we have to try to give the Ukrainians more of a technological advantage," he noted.

The United States has said that it will begin flight training for Ukrainian pilots on F-16 fighter jets. Anadolu Agency | Getty Images

Volker believes that a Trump presidency might not be the catastrophe for Ukraine that is feared, but said it would make future funding uncertain. "I doubt that
even if Trump were elected that he would abandon support for Ukraine overall, because it would be a disaster for U.S. interests, and it would appear to be a
failure. You'd have these images of Russians overrunning places, and brutality and so forth, so I don't think he wants that. But it's not clear exactly what he
would do to try to end the war."

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Getty Images
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Getty Images
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Getty Images
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Getty Images
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Getty Images

Entities: Getty Images, Russia, Ukraine, war, United States, funding, Trump, CNBC, With hopes of victory fading, Ukraine's fight against Russia could get even
harder in 2024, F-16 fighter jets, Volker, Ukrainian pilots, Anadolu Agency, Trump presidency, U.S. interests
</item 5>

<item 6>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
The twenty-first century is also about the struggle against new and rising forms of authoritarianism. In this narration, the twenty-first century began with a
wave of crushed democratic uprisings and continued with the election of authoritarian leaders around the world who began to dismantle democratic institutions.
Any illusion of the success of globalization, or of the twenty-first century representing a break from the brutal twentieth century, was stripped away with
Russia’s most recent invasion of Ukraine. The plot is less clear, given the failure of democracy-building efforts in the twentieth century. There is a faintly
discernable storyline of general resistance and rebuilding imperfect democracies.

There’s also a third story about this century: the penetration of the Internet into every sphere of daily, social, and political life. Despite
turn-of-the-century talk about the Information Age, we are only beginning to conceptualize what this means. Right now, the current plot is about the
centralization of discourse on a few corporate platforms.

Graph facts:
- Dissecting impact of Israel-Hamas conflict on markets -MENTIONS-> Russia
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> Russia
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> Russia
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Russia
- With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024 -MENTIONS-> Russia

Entities: Russia, internet, Ukraine, globalization, Authoritarianism, democratic institutions, democracy-building efforts, Information Age, corporate platforms,
Russia’s most recent invasion of Ukraine, Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it., Engadget
</item 6>

<item 7>
With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024 (Cnbc | World Business News Leader, 2023-12-25T23:36:00+00:00)
A Ukrainian soldier is seen inside an artillery vehicle in his fighting position as Russia-Ukraine war continues in the direction of Kharkiv, Ukraine on
November 20, 2023. Diego Herrera Carcedo | Anadolu | Getty Images

At the start of 2023, hopes were high that a much-vaunted Ukrainian counteroffensive — expected to be launched in the spring — would change the dial in the war
against Russia. It didn't, and the prospect of a breakthrough in 2024 is also unlikely, military experts and defense analysts told CNBC. They predict intense
fighting is likely to continue into the next year but say Kyiv's forces are unlikely to launch any more counteroffensives. Russia, meanwhile, is likely to focus
on consolidating the territory it has already seized, particularly in eastern Ukraine.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Getty Images
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Getty Images
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Getty Images
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Getty Images
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Getty Images

Entities: Getty Images, Russia, Ukraine, CNBC, Ukrainian counteroffensive, Kyiv, Kharkiv, Diego Herrera Carcedo, Anadolu, Russia-Ukraine war, With hopes of
victory fading, Ukraine's fight against Russia could get even harder in 2024
</item 7>

<item 8>
With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024 (Cnbc | World Business News Leader, 2023-12-25T23:36:00+00:00)
With both Ukraine and Russia investing heavily in the war, it's unlikely there will be any negotiations to end the conflict or agree to a cease-fire. Defense
analysts argue that neither side would want to go into negotiations unless they're in a position of strength and able to dictate terms. "In the case of a
Republican winning the presidential election next year, especially if that's Donald Trump, who seems to be the front-runner, and [if] funding is decreased
substantially, then there will be increased pressure on Ukraine to negotiate," Mario Bikarski, a Europe and Russia analyst at the Economist Intelligence Unit,
told CNBC.

A Ukrainian tank drives along the field on December 7, 2023 in Avdiivka, Ukraine. Kostya Liberov | Getty Images

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Getty Images
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Getty Images
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Getty Images
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Getty Images
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Getty Images

Entities: Getty Images, CNBC, With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024, Avdiivka, Ukraine, Russia, war, Mario
Bikarski, Economist Intelligence Unit, Kostya Liberov, presidential election, Donald Trump
</item 8>

<item 9>
With hopes of victory fading, Ukraine's fight against Russia could get even harder in 2024 (Cnbc | World Business News Leader, 2023-12-25T23:36:00+00:00)
Former U.S. Ambassador to NATO Kurt Volker told CNBC he believes American and EU aid packages for Ukraine will be approved come January, saying he believed this
funding would tide Ukraine over for another year, militarily. Volker said that aid packages must include more advanced weaponry for Ukraine, however, like F-16
fighter jets which have been pledged by Norway, Denmark and the Netherlands. Ukrainian pilots are beginning their training on the jets now but it could be a
number of months before they're deployed in Ukraine. The U.S. is not providing F-16s to Ukraine but has authorized allies to provide their own jets. "A couple
of things ought to change," Volker told CNBC. "We ought to lift restrictions on the weapons we're providing. We still don't provide the longest-range missiles
and we still have not delivered any Western aircraft in Ukraine yet. Those things have to happen.

Graph facts:
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> EU
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> EU
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> EU
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> EU
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> EU

Entities: EU, Denmark, Norway, Russia, Ukraine, CNBC, U.S., Netherlands, NATO, With hopes of victory fading, Ukraine's fight against Russia could get even
harder in 2024, Kurt Volker, F-16 fighter jets, American and EU aid packages, Advanced weaponry, Longest-range missiles
</item 9>

<item 10>
Graph query row:
article: Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it.
relationship: MENTIONS
connected_entity: Ukraine
</item 10>
=== END ITEM id=35962659466f7d09 ===

=== ITEM id=37fd4074bc0a4c06 ===
Gold sentences and facts:
Sentence 0: Because FTX had grown so much, he could no longer run both companies, he said.
  Fact 0.0: FTX had grown so much.
  Fact 0.1: The speaker said that because FTX had grown so much, he could no longer run both companies.
Sentence 1: And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.
  Fact 1.0: After Bankman-Fried left Jane Street, he persuaded Yedidia to join him as a trader at Alameda.
  Fact 1.1: Bankman-Fried left Jane Street.
  Fact 1.2: Jane Street is a high-frequency trading firm.
  Fact 1.3: Bankman-Fried got his start in finance at Jane Street after MIT.
  Fact 1.4: Bankman-Fried is a former billionaire.
  Fact 1.5: Bankman-Fried later persuaded Yedidia to join him as a developer at FTX.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, The jury finally hears from Sam Bankman-Fried, Alameda Research, FTX, Nishad Singh, Wang, Sam Bankman-Fried, The Verge
</item 1>

<item 2>
Graph query row:
article: Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand
published_on: 2023-10-04T23:32:53+00:00
publisher: Fortune
article_relationship: MENTIONS
mentioned_entity: Adam Yedidia
entity_relationship: MENTIONS
connected_entity: Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand
</item 2>

<item 3>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, The jury finally hears from Sam Bankman-Fried, Alameda, Sam Trabucco, MobileCoin loss, Singh, FTX, Wang, Sam Bankman-Fried,
MobileCoin, The Verge
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Gary Wang

Entities: Gary Wang, Securities and Exchange Commission, misappropriation of customer funds, Alameda, Christian Everdell, spot margin program, Sam Bankman-Fried
was a terrible boyfriend, FTX, Sam Bankman-Fried, trial, fraud, The Verge
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
It was that Ellison’s calculations assumed Alameda could borrow $1.8 billion in normal dollars and $1.5 billion in crypto from FTX. The spreadsheet makes this
clear with a row labeled “FTX borrows,” which Ellison said were customer funds.

Meanwhile, echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length. On
January 14th, 2022, Bankman-Fried tweeted, “We’re launching a $2b venture fund: FTX Ventures!” Those funds came from Alameda, Ellison testified. But
Bankman-Fried didn’t want to go public with the source of the funds. He said he thought Alameda’s brand was less good, and he didn’t want his name associated
with it. Alameda also bought Robinhood shares for Bankman-Fried, who moved them to a vehicle called “Emergent Fidelity Technologies” to avoid association with
Alameda.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> crypto
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> crypto
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> crypto

Entities: crypto, Alameda, Sam Bankman-Fried was a terrible boyfriend, FTX Ventures, Emergent Fidelity Technologies, Ellison, customer funds, FTX, Sam
Bankman-Fried, Robinhood, The Verge
</item 5>

<item 6>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

Graph facts:
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Securities and Exchange Commission
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> Securities and Exchange Commission
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> misappropriation of customer funds

Entities: Securities and Exchange Commission, misappropriation of customer funds, Sam Bankman-Fried trial, wire fraud, Ellison, FTX, Sam Bankman-Fried,
conspiracy to take FTX customer funds, fake balance sheets, How is it still getting worse for Sam Bankman-Fried?, Genesis, The Verge
</item 6>

<item 7>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Sam Bankman-Fried was a terrible boyfriend, FTT, Ellison, FTX, Wang, Sam Bankman-Fried, Binance, The Verge
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, FTX trial, The FTX trial is bigger than Sam Bankman-Fried, extensive media tour, interviews, Andrew Ross Sorkin, Good Morning
America, The New Yorker, FTX, Sam Bankman-Fried, The Verge
</item 8>

<item 9>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> crypto
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> crypto
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> crypto
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto

Entities: crypto, Alameda, FTX trial, second trial, misappropriated FTX customer deposits, political contributions, The FTX trial is bigger than Sam
Bankman-Fried, real estate, FTX, Sam Bankman-Fried, The Verge
</item 9>

<item 10>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Damian Williams
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The best new TV series on every streaming service -MENTIONS-> Ponzi scheme
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Matt Huang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison

Entities: Ponzi scheme, crypto markets, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand,
Alameda, Ellison, Cohen, Fortune, FTX, Sam Bankman-Fried
</item 10>
=== END ITEM id=37fd4074bc0a4c06 ===

=== ITEM id=1b3cb3874b494a9c ===
Gold sentences and facts:
Sentence 0: “The product 6PPD-quinone, which was identified in late 2020, requires more research to understand what impact, if any, it may have on aquatic life or the environment,” Goodyear SVP of global operations and CTO Chris Helsel says.
  Fact 0.0: The product 6PPD-quinone was identified in late 2020.
  Fact 0.1: The product 6PPD-quinone requires more research to understand what impact, if any, it may have on aquatic life or the environment.
  Fact 0.2: Chris Helsel is SVP of global operations and CTO at Goodyear.
  Fact 0.3: Chris Helsel says this.
Sentence 1: In our report, a former VP at an industrial automation vendor shared, “Digital solutions provide the visibility, analysis, and insight needed to address the challenges inherent in sustainability goals.
  Fact 1.0: A former VP at an industrial automation vendor shared a view in the report.
  Fact 1.1: The former VP said digital solutions provide the visibility, analysis, and insight needed to address the challenges inherent in sustainability goals.

Retrieved items:
<item 1>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
New research aims to improve the stability and safety of alternatives to rechargeable lithium-ion batteries using aqueous zinc and hydrogels.

The demand for rechargeable batteries as a result of the growth of both the personal electronics and electric vehicle markets has grown steadily over the last
decade. Lithium-ion batteries are one of the technologies that have emerged and matured due to this demand.

In 2020, Statista estimated that lithium-ion batteries accounted for 185 gigawatts (GWh) of energy consumed, and this energy demand is projected to reach 2,035
GWh by 2030. Most of this growth is accounted for by the increased desire for electric vehicles as people turn away from fossil fuels.

While it is undeniable that lithium-ion batteries are more environmentally friendly than fossil fuels and while they are also relatively inexpensive, these
batteries aren’t perfect.

Graph facts:
- Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market -MENTIONS-> Environmental impact
- Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market -MENTIONS-> sustainability goals
- Microsoft Activision: What does deal mean for gamers? -MENTIONS-> Statista
- Can anyone survive Fortnite as a job? -MENTIONS-> Statista
- Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market -MENTIONS-> Advanced Materials

Entities: aqueous rechargeable zinc iodine batteries, stability and safety, rechargeable lithium-ion batteries, rechargeable batteries, personal electronics
market, electric vehicle markets, Statista, aqueous zinc, hydrogels, Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market,
Advanced Science News, electric vehicles, fossil fuels
</item 1>

<item 2>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
This will include optimizing the production process of hydrogels as well as sourcing affordable raw materials and minimizing waste generated during the
manufacturing of these materials.

“Ensuring the safety and reliability of hydrogel-based batteries is also necessary,” Fan concluded. “Rigorous testing and evaluation of the potential
environmental impact of hydrogel electrolytes, from raw material extraction to disposal, are vital to meeting sustainability goals and gaining public trust.”

Reference: Peihua Yang, Hong Jin Fan, et al., Hetero-Polyionic Hydrogels Enable Dendrites-Free Aqueous Zn-I2 Batteries with Fast Kinetics, Advanced Materials,
(2023). DOI: 10.1002/adma.202306531

Feature image credit: danilo.alvesd on Unsplash

Graph facts:
- A new hydrogel could help pull drinking water from the air -MENTIONS-> Advanced Materials
- Molden -SAID-> Environmental impact
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> sustainability goals
- Tires are saving us — and killing us, too -MENTIONS-> Environmental impact
- A new hydrogel could help pull drinking water from the air -MENTIONS-> hydrogels

Entities: Environmental impact, sustainability goals, Advanced Materials, hydrogels, aqueous rechargeable zinc iodine batteries, Bringing aqueous rechargeable
zinc iodine batteries to the mainstream energy market, Fan, Hetero-Polyionic Hydrogels Enable Dendrites-Free Aqueous Zn-I2 Batteries with Fast Kinetics, Peihua
Yang, Hong Jin Fan, hydrogel electrolytes, safety and reliability of hydrogel-based batteries, public trust, Advanced Science News
</item 2>

<item 3>
Graph query row:
article: Tires are saving us — and killing us, too
relationship: MENTIONS
related_entity: 6PPD-quinone
</item 3>

<item 4>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
If not, what investments are required?

2. Data integration: How compatible are our current data sources and systems with digital twin technologies, especially in terms of sensor data (both physical
and virtual/soft)?

3. Interoperability concerns: As we use multiple software solutions across our operations, how important is interoperability between digital twin technologies
and our existing systems?

Graph facts:
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> software
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Microsoft
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> China
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> digital twin market
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Digital Twin Market Report 2023–2027

Entities: IoT Business News, Digital twin market: Analyzing growth and 4 emerging trends, digital twin technologies, data integration, sensor data, software
solutions, investments, interoperability
</item 4>

<item 5>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
“What you don’t want is all these little wear particles full of all these nasty compounds settling in the environment and inside us and gradually leaching out
their toxins,” says Nick Molden, CEO of Emissions Analytics, an independent group focused on testing and data about the overall environmental impacts of
transportation.

“What you don’t want is all these little wear particles full of all these nasty compounds settling in the environment and inside us and gradually leaching out
their toxins.”

One such toxin is 6PPD-quinone. A 2020 study from the University of Washington’s Center for Urban Waters found that a dramatic decline in coho salmon
populations was the result of the toxin. The salmon have now been placed on some West Coast endangered lists.

“Shortly after birth, they were essentially going mad and dying,” Molden says. “The researchers finally linked it to this compound in tires because it was
happening after rainfall. The rainfall was washing all the accumulated tire material off the road into the harbors and into the shore.”

Graph facts:
- Tires are saving us — and killing us, too -MENTIONS-> Imperial College London
- Bridgestone -MADE-> tires
- Harrigle -SAID-> tires
- Tires are saving us — and killing us, too -MENTIONS-> LinkedIn
- Tires are saving us — and killing us, too -MENTIONS-> tire wear particles

Entities: tires, Tires are saving us — and killing us, too, Nick Molden, Emissions Analytics, University of Washington, Center for Urban Waters, 2020 study,
6PPD-quinone, coho salmon, West Coast, environmental impacts of transportation, The Verge
</item 5>

<item 6>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
“For the next five years, if you want to sell tires in California, you need to go through the process of either eliminating 6PPD, or at least properly
researching the alternatives,” Molden says.

“We are collaborating with researchers and other scientists to better understand this transformation product, fill knowledge gaps, and determine next steps,”
says Helsel.

Other initiatives are coming to limit overall tire emissions, including the upcoming Euro 7 regulations, which put caps on tire wear particles as well as other
emissions.

That seems like progress, but Emissions Analytics’ Molden isn’t so sure: “The reason I’m worried about that is I’m sure you can reduce that wear mass, but if
you do that by making the compound more toxic, you could actually make the environmental impact worse.”

“You could actually make the environmental impact worse.”

That could be akin to increasing a car’s fuel efficiency by switching to leaded fuel.

Molden believes that what’s needed is better understanding.

Graph facts:
- Tires are saving us — and killing us, too -MENTIONS-> Imperial College London
- Bridgestone -MADE-> tires
- Harrigle -SAID-> tires
- Tires are saving us — and killing us, too -MENTIONS-> LinkedIn
- Tires are saving us — and killing us, too -MENTIONS-> tire wear particles

Entities: tires, Tires are saving us — and killing us, too, Emissions Analytics, Molden, 6PPD, Euro 7 regulations, Tire emissions, Tire wear particles, Leaded
fuel, Environmental impact, Fuel efficiency, Helsel, California, car, The Verge
</item 6>

<item 7>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
That isn’t bad just for the salmon and the Pacific Northwest’s fishing industry. 6PPD-quinone is in us, too, and it only comes from tires. 6PPD helps tires
resist environmental damage, thus lasting longer between replacements. When 6PPD reacts with ozone, it becomes 6PPD-quinone.

“The product 6PPD-quinone, which was identified in late 2020, requires more research to understand what impact, if any, it may have on aquatic life or the
environment,” Goodyear SVP of global operations and CTO Chris Helsel says. That’s hardly a mea culpa, but the tire industry genuinely has been an active partner
in studying the problem.

Goodyear and other members of the US Tire Manufacturers Association actually encouraged 6PPD-targeting regulation enacted by the California Environmental
Protection Agency and have provided free tire compound samples to researchers.

Graph facts:
- Tires are saving us — and killing us, too -MENTIONS-> Imperial College London
- Bridgestone -MADE-> tires
- Harrigle -SAID-> tires
- Tires are saving us — and killing us, too -MENTIONS-> LinkedIn
- Tires are saving us — and killing us, too -MENTIONS-> tire wear particles

Entities: tires, Tires are saving us — and killing us, too, Chris Helsel, Goodyear, US Tire Manufacturers Association, California Environmental Protection
Agency, 6PPD-quinone, 6PPD, 6PPD-targeting regulation, The Verge
</item 7>

<item 8>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Considerations for digital twin vendors

3 questions digital twin vendors should ask themselves based on research findings discussed in this article:

1. Sustainability goals: Given the emphasis on sustainability and the role of digital twins in achieving it, is my company’s digital twin solution well-equipped
to support sustainability objectives? Are there new features we need to develop?

2. Interoperability: Is our digital twin solution currently designed to be interoperable with other systems and technologies, especially those from different
vendors?

3. Product development: How aligned is our product development roadmap with the emerging trends highlighted in the report, especially around end-user and
technological advancements?

Considerations for adopters

3 questions adopters should ask themselves based on research findings discussed in this article:

1. Infrastructure readiness: Do we have the necessary infrastructure and resources in place to support the deployment and effective use of digital twins? If
not, what investments are required?

2.

Graph facts:
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> software
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Microsoft
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> China
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> digital twin market
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Digital Twin Market Report 2023–2027

Entities: IoT Business News, Digital twin market: Analyzing growth and 4 emerging trends, digital twin solution, product development, infrastructure readiness,
sustainability, interoperability
</item 8>

<item 9>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Digital twin market trends

We will now look at four trends helping digital twin market growth.

Trend 1: Digital twins deployed to meet sustainability goals

Sustainability was discussed in approximately 21% of recent CEO earnings calls and has remained a consistent topic throughout 2023. This pairs with our report’s
analysis that the pursuit of sustainability goals is a tailwind macro factor for the digital twin market.

To achieve their sustainability goals, many companies are exploring digital twins. Due to their ability to simulate real-world conditions and deliver real-time
information, organizations can optimize resource usage, reducing carbon emissions and improving supply and transportation networks.

According to Capgemini Research Institute’s 2022 digital twins report, 57% of organizations agreed that one of the key drivers for their digital twin
investments was improving their sustainability, and 51% agreed that digital twins would help achieve their organization’s environmental sustainability goals.

In our report, a former VP at an industrial automation vendor shared, “Digital solutions provide the visibility, analysis, and insight needed to address the
challenges inherent in sustainability goals.

Graph facts:
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Microsoft
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> China
- IoT Analytics -SAID-> digital twins
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> software
- Digital twin market: Analyzing growth and 4 emerging trends -MENTIONS-> Digital Twin Market Report 2023–2027

Entities: IoT Business News, digital twin market, digital twins, Capgemini Research Institute, 2022 digital twins report, Digital twin market: Analyzing growth
and 4 emerging trends, sustainability
</item 9>

<item 10>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Truly sustainable tires are becoming a reality, but will it be enough?

Renewable rubber

Bridgestone’s “Wireless-In-Wheel” technology.

If you watched any IndyCar road course race this year, basically any of the half-dozen events where the drivers had to turn right as well as left, you might
have noticed some green flashes on those cars’ tires. It wasn’t just a visual flair to an otherwise visually dull part of the equation.

Those tires contained guayule, part of an agrarian prototype effort from Bridgestone, the supplier of spec tires for that series. It’s a high-profile
demonstration of a new domestically sourced, renewable source of rubber that augments other natural sources like that from the hevea tree and even the Russian
dandelion.

Truly sustainable tires are becoming a reality, but will it be enough?

Bridgestone’s Bill Niaura, executive director of sustainable innovation and circular economy, says guayule is a drop-in replacement with many potential
advantages. Chief among them?

Graph facts:
- Bridgestone -MADE-> Turanza EV
- Niaura -WORKS_FOR-> Bridgestone
- Dale Harrigle -WORKS_FOR-> Bridgestone
- Bridgestone -PARTICIPATED_IN-> Japan Mobility Show
- Harrigle -SAID-> tires

Entities: Bridgestone, Bill Niaura, IndyCar, tires, Wireless-In-Wheel, guayule, renewable rubber, hevea tree, Russian dandelion, Tires are saving us — and
killing us, too, The Verge
</item 10>
=== END ITEM id=1b3cb3874b494a9c ===
