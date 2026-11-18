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

=== ITEM id=d1d6154ffc87e025 ===
Gold sentences and facts:
Sentence 0: For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.
  Fact 0.0: The victory has the Swans challenging for a top-eight spot.
  Fact 0.1: The Swans are challenging for a top-eight spot alongside teams such as Gold Coast.
  Fact 0.2: The Swans are challenging for a top-eight spot alongside teams such as Geelong.
  Fact 0.3: The Swans are challenging for a top-eight spot alongside teams such as Collingwood.
  Fact 0.4: The Swans are challenging for a top-eight spot alongside teams such as St Kilda.
  Fact 0.5: The Swans are challenging for a top-eight spot alongside teams such as Carlton.
Sentence 1: With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot.
  Fact 1.0: Only 0.4 of a percentage point separates Sydney in ninth place and St Kilda in eighth place.
  Fact 1.1: Sydney is in ninth place.
  Fact 1.2: St Kilda is in eighth place.
  Fact 1.3: The win assured the Swans of retaking eighth spot.
Sentence 2: “Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting our games,” she said.
  Fact 2.0: A speaker, referred to as 'she', said the following quote.
  Fact 2.1: The speaker's team has put a lot of work into sourcing some wonderful food offerings.
  Fact 2.2: The speaker's team has put a lot of work into sourcing entertainment for kids.
  Fact 2.3: The speaker's team has put a lot of work into sourcing activities for kids.
  Fact 2.4: The speaker's team has had a local focus in promoting its games.

Retrieved items:
<item 1>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Fans of Australian rules football in the Harbour City have a reputation for being fickle. After a winless campaign in 2022, the Swans – led by ex-Collingwood
player Chloe Molloy – have been competitive this season. The team won their first AFLW match in round one and are within a game of the top eight with two rounds
to play.

But general manager of women’s football at the AFL, Nicole Livingstone, said the Swans’ off-field offering had tapped into something “uniquely Sydney” and
helped make matches more attractive.

“To go to North Sydney Oval and to have quince paste and hand-picked grapes, cheddar cheese and biscuits in a little box that’s got this beautiful wrapping on
it – they’ve really thought about what they’re doing and made it an experience,” she said.
</item 1>

<item 2>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“Even at Henson Park [the Swans’ other home ground in Sydney’s gentrified inner west], they’ve gone to microbreweries in the local area and invited them in, but
also promoted through their clientele to come to the football.”

The Swans forward Bella Smith said the players were proud of the club’s attendance record, and were grateful for the support. “We get there two hours before the
start of the game and the hill [at Henson Park] is bare. When you come out for the first team warm-up, the hill is slowly filling and then by the time the first
bounce gets around, it is full,” she said. “You can hear it, the atmosphere is amazing.”

The average attendance for the AFLW is a little over 2,500 a match this year, tracking approximately on par with previous seasons and with a finals uptick
likely to come.
</item 2>

<item 3>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Down by four points at half-time in the low-scoring game, a third-term goal from Rebecca Privitelli sparked the Swans to a 5.10 (40) to 2.6 (18) win on Sunday
at Fremantle Oval.

The second-year expansion side stormed home with three unanswered goals to secure eighth spot. It’s a dramatic improvement, with the Swans winless in their
first season. “Any club that’s able to go from winless one year to a finals berth, it’s incredible,” Sydney co-captain Chloe Molloy told Fox Footy after the
match. “You can see the emotion on the girls’ faces. Coming into the season, the odds were against us. Just to get into the eight, we’ve managed to do that.
We’re stoked. It’s pretty special for the club.”
</item 3>

<item 4>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But Livingstone said the AFL was happy with the level of competition. “Our ladder, from position five to 13 – there’s only a game separating it. It just shows
that this is a really competitive competition.”
</item 4>

<item 5>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“We’ll continue to keep building the base and keep building the foundations and have it not just strong at year seven, season eight, but strong for generations
to come.”

In a bid to improve the spectacle of the AFLW, the league introduced longer quarters this year and an interchange cap to help increase scoring. The effect,
according to Livingstone, has been positive.

“We’ve increased game time by eight minutes per match, on average, and we’ve had record scores across the rounds,” she said. “Whereas leading into the season we
hadn’t scored more than 100 goals in a round, we’ve regularly been scoring over 100 goals in a round.”

The Swans recorded their highest total last round, in their 14-12-96 to 6-3-39 defeat of the Western Bulldogs, the league’s winless bottom side. Molloy recorded
17 score involvements to set a new AFLW mark.

But Livingstone said the AFL was happy with the level of competition.
</item 5>

<item 6>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Quince paste and hand-picked grapes have played a part in helping the Sydney Swans launch themselves to the top of the AFLW attendance charts, less than two
years after their first match in the women’s league.

The Swans play their final home match of the regular season this weekend with an average attendance of 4,365 from four games so far – 124 more than the next
best club Adelaide – and a marquee pride match against Collingwood at Henson Park to come on Sunday where a crowd in excess of 4,000 is expected.

The club’s focus has been to target local communities and make match days compelling for fans on and off the field, according to executive GM of marketing at
the Swans, Loretta Johns.

“Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting
our games,” she said.

Fans of Australian rules football in the Harbour City have a reputation for being fickle.
</item 6>

<item 7>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot. If the Saints had
qualified, it would have been their maiden finals appearance. Saints goalkicker Jesse Wardlaw had a heavy collision with Carlton’s Keeley Sherar late in the
final quarter of their clash.

Georgia Patrikios showed her class in the do-or-die match, collecting 25 disposals and eight contested possessions in a powerhouse display, while Hannah Priest
was a brick wall in defence, with nine intercept possessions. Abbie McKay worked overtime to counter Patrikios in the midfield, but her side was outclassed
across the park. McKay racked up 28 disposals and 12 clearances alongside Mimi Hill (30 disposals), while defender Sherar (17) toiled against a potent Saints
attack. Meanwhile, Sydney prevailed in a fierce tug-of-war with the Dockers to seal their first AFLW finals appearance.
</item 7>

<item 8>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 8>

<item 9>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.
</item 9>

<item 10>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
Molloy, the Swans’ star off-season recruit, turned on one of the best performances of her career to demolish their injury-hit hosts.

The former Collingwood ace was a class above as she kicked 3.4 from her 28 possessions and played a crucial role in setting up numerous other Sydney goals.
Loading “She [Molloy] was itching to get in there [in the midfield],” Swans coach Scott Gowans said. “We kept her forward deliberately. It was important Chloe
played a good block of time forward so we could have a look at it.” The Blues lost to Greater Western Sydney by two points on Saturday afternoon, denying them
the chance to leapfrog into the top eight, 8.1 (49) to 7.5 (47).

Narrm reclaim top spot Reigning premiers Narrm have reclaimed top spot on the ladder and their position as flag favourites after defeating third-placed North
Melbourne by 23 points at Ikon Park on Saturday night.
</item 10>
=== END ITEM id=d1d6154ffc87e025 ===

=== ITEM id=55f05337cfb6d1a4 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that 'all of that, all of it, was built on lies.'
Sentence 1: And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.
  Fact 1.0: Bankman-Fried left Jane Street.
  Fact 1.1: Jane Street is a high-frequency trading firm.
  Fact 1.2: Bankman-Fried was a former billionaire.
  Fact 1.3: Bankman-Fried got his start in finance at Jane Street.
  Fact 1.4: Bankman-Fried got his start in finance after MIT.
  Fact 1.5: After leaving Jane Street, Bankman-Fried persuaded Yedidia to join him as a trader at Alameda.
  Fact 1.6: Bankman-Fried later persuaded Yedidia to join him as a developer at FTX.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that the FTX founder, Bankman-Fried, acted in good faith.
  Fact 2.2: The defense said Bankman-Fried never meant to commit fraud or steal.
  Fact 2.3: The defense said Bankman-Fried basically got in over his head.

Retrieved items:
<item 1>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.
</item 1>

<item 2>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The recording was later sent to Drappi, who sent it to the government. Joseph Bankman, the defendant’s father and a senior advisor to FTX’s philanthropic arm,
wasn’t in the room for the recording. Barbara Fried looked unhappy, rubbing her left temple as though she had a tremendous headache.

And in a small miracle of pacing, the government got Zac Prince, the founder and former CEO of crypto lender BlockFi, on the stand just long enough to blame
FTX’s bankruptcy for BlockFi’s subsequent bankruptcy before we broke for the day.

I’ve been asked by some people why the reporting coming out of the trial seems so skewed toward the prosecution. There’s an answer for this: the prosecution has
put together a strong, comprehensive indictment of Bankman-Fried’s behavior at FTX and Alameda. The defense, so far, has managed to do almost nothing in
response.
</item 2>

<item 3>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 4>

<item 5>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
In the indictment, the prosecutors allege that Bankman-Fried lied to FTX investors. Some of those investors may be called on to testify, which is not the kind
of thing VC big shots generally enjoy — and which might create collateral damage for the industry.

If, for instance, Sequoia Capital did due diligence around its investment in FTX, whatever Bankman-Fried told its partners could be important. Was FTX already
sending Alameda customer funds at that point? Did Bankman-Fried know about it? And did he tell VCs at the time? (If he told them it wasn’t happening, that would
establish intent.) Binance was also an early investor in FTX, and former executives may be called on to testify as well, says Hermine Wong, the former head of
policy at Coinbase and a former SEC regulator.

Bankman-Fried’s defense can also introduce risks for people who dealt with him. Defense lawyers have several simultaneous objectives. First and foremost,
they’re trying to prove their client is not guilty.
</item 5>

<item 6>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 6>

<item 7>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
I have never seen Sam Bankman-Fried so still as he was during the prosecution’s opening statement. The characteristic leg-jiggling was absent. He barely moved
as the prosecutor listed the evidence against him: internal company files, what customers were told, the testimony of his co-conspirators and his own words.

His hair was shorn, the result of a haircut from a fellow prisoner, the Wall Street Journal reported. He wore a suit bought at a discount at Macy’s, per the
Journal; it hung on him. He appeared to have lost some weight.

“All of that was built on lies.”

Bankman-Fried, at this time last year, had a luxury lifestyle as the CEO of crypto exchange FTX, said the assistant US attorney, Thane Rehn, in the cadence of a
high schooler delivering his lines in a student play. Bankman-Fried hung out with Tom Brady. He was on magazine covers, lived in a $30 million penthouse, and
spent time with world politicians.
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.
</item 8>

<item 9>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.
</item 10>
=== END ITEM id=55f05337cfb6d1a4 ===

=== ITEM id=914d4df2487f4d1c ===
Gold sentences and facts:
Sentence 0: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.
  Fact 0.0: SBF is known as SBF.
  Fact 0.1: SBF has pleaded not guilty to some seven charges.
  Fact 0.2: The charges are of fraud and conspiracy.
Sentence 1: Because FTX had grown so much, he could no longer run both companies, he said.
  Fact 1.0: FTX had grown so much.
  Fact 1.1: SBF said he could no longer run both companies because FTX had grown so much.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense countered that the FTX founder acted in good faith.
  Fact 2.2: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Perhaps you are wondering, was the jury also there? Reader, it was.

Bankman-Fried denied he knew about the effectively infinite line of credit Alameda Research received from FTX. This argument was peculiar; essentially my
take-away was that the CEO of a financial company simply didn’t pay attention to finances.

FTX couldn’t get bank accounts right away. Bankman-Fried anticipated it would take a year or two. Rather than wait, he decided to use Alameda as the “payment
provider” for bank transfers. “My understanding at the time was that there were teams managing the process,” he said. “At the time, I wasn’t entirely sure what
was happening.”

Well, sure, understandable! He’s an introvert!

Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the line” to $1 billion

In 2021, FTX was growing to millions of users, with $1 billion of revenue.
</item 1>

<item 2>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
The prosecution painted a picture of an empire “built on lies” and made a point to show actual pictures of SBF alongside rich and powerful figures to illustrate
the peak of his power and influence prior to FTX’s implosion.

“This man stole billions of dollars from thousands of people,” prosecutor Thane Rehn told the court. “He defrauded sophisticated investors and lenders, and he
emptied the accounts of ordinary customers, too. He bought himself wealth, power and influence.”

The defense, on the other hand, depicted SBF and his colleagues as in over their heads, without ill-will as FTX collapsed around them.

“Sam and his colleagues were building the plane as they were flying it,” defense attorney Mark Cohen said. “They had to figure out how to navigate a world where
they were running FTX, building out its systems, dealing with hacking threats, managing the credit risk of their customers, managing hundreds of employees, all
while building up their actual exchange.”

Read a recap of the prosecution and defense’s full arguments here.
</item 2>

<item 3>
Graph query row:
title: The jury finally hears from Sam Bankman-Fried
date: 2023-10-28T00:12:41+00:00
publisher: The Verge
connectedFacts: [{'entity': 'Sam Bankman-Fried', 'related': 'Alameda Research', 'relationship': 'MENTIONS'}, {'entity': 'Sam Bankman-Fried', 'related':
'Alameda', 'relationship': 'MENTIONS'}, {'entity': 'Sam Bankman-Fried', 'related': 'FTX', 'relationship': 'MENTIONS'}, {'entity': 'Sam Bankman-Fried',
'related': 'Alameda Research', 'relationship': 'SAID'}, {'entity': 'Sam Bankman-Fried', 'related': 'Alameda', 'relationship': 'SAID'}, {'entity': 'Sam
Bankman-Fried', 'related': 'FTX', 'relationship': 'SAID'}, {'entity': 'Sam Bankman-Fried', 'related': 'Alameda Research', 'relationship': 'WORKS_FOR'},
{'entity': 'Sam Bankman-Fried', 'related': 'Alameda', 'relationship': 'WORKS_FOR'}, {'entity': 'Sam Bankman-Fried', 'related': 'FTX', 'relationship':
'WORKS_FOR'}, {'entity': 'Sam Bankman-Fried', 'related': 'Alameda Research', 'relationship': 'OWNS'}, {'entity': 'Sam Bankman-Fried', 'related': 'Alameda
Research', 'relationship': 'FOUNDED'}, {'entity': 'Sam Bankman-Fried', 'related': 'FTX', 'relationship': 'FOUNDED'}]
</item 3>

<item 4>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
The full breakdown of his testimony on his relationship with FTX, stretching back to 2019, can be found right here.

We also got a look at the composition of the SBF trial’s jury, which ranges from a train conductor to a retired investment banker who attended Stanford
University. You can listen to a special joint episode of our podcasts Equity and Chain Reaction to get a better sense of the vibe between the jurors and the
rest of the court.

For a full rundown of our coverage of the SBF trial, check below:
</item 4>

<item 5>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 5>

<item 6>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
You can also follow along with her and the TechCrunch team’s coverage in the Chain Reaction newsletter, which drops Thursdays at 12 p.m. PT, and the Chain
Reaction podcast, which will hit your feed every Thursday.

How to follow the SBF trial

The trial kicked off with jury selection October 3. From there, access to the proceedings is limited, as no devices are allowed; reports are being physically
put together within the courtroom (or sent to overflow rooms).

Beyond our daily coverage, additional expertise and commentary will live on TechCrunch+. There, you can find stories like this in-depth breakdown of what to
expect from both sides of the SBF case, where the prosecution and defense could gain ground or fall short in their arguments and what the takeaways were from
the opening arguments.

SBF trial: What we learned in week three

The third week of the trial featured a couple of noteworthy moments for the prosecution, including FTX’s former general counsel Can Sun.
</item 6>

<item 7>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 7>

<item 8>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
Read a recap of the prosecution and defense’s full arguments here.

FTX’s co-founder and CTO Gary Wang testified Thursday on the extent to which Alameda Research was used for whatever purposes he and SBF saw fit. The crypto
trading firm, according to Wang, pulled funding directly from FTX customers, whose transactions would be funneled toward Alameda and then directed elsewhere.

Check here for our rundown of Wang’s testimony so far, resulting from a guilty plea in December 2022.

Thursday also brought in testimony from Matt Huang, co-founder and managing partner at crypto investment firm Paradigm. That firm invested $278 million in FTX
across 2021 and 2022, and is part of a class-action lawsuit accusing it and others of defrauding customers by promoting FTX.

His testimony centered around being left uninformed about FTX’s utilization of customer money to keep the Alameda Research afloat, a practice that Huang says
would have halted his involvement.
</item 8>

<item 9>
Graph query row:
title: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
date: 2023-10-06T17:22:02+00:00
publisher: TechCrunch
connectedFacts: [{'entity': 'Sam Bankman-Fried', 'related': 'FTX Ventures', 'relationship': 'MENTIONS'}, {'entity': 'Sam Bankman-Fried', 'related': 'Alameda
Research', 'relationship': 'MENTIONS'}, {'entity': 'Sam Bankman-Fried', 'related': 'Alameda', 'relationship': 'MENTIONS'}, {'entity': 'Sam Bankman-Fried',
'related': 'FTX', 'relationship': 'MENTIONS'}, {'entity': 'Sam Bankman-Fried', 'related': 'Alameda Research', 'relationship': 'SAID'}, {'entity': 'Sam
Bankman-Fried', 'related': 'Alameda', 'relationship': 'SAID'}, {'entity': 'Sam Bankman-Fried', 'related': 'FTX', 'relationship': 'SAID'}, {'entity': 'Sam
Bankman-Fried', 'related': 'Alameda Research', 'relationship': 'WORKS_FOR'}, {'entity': 'Sam Bankman-Fried', 'related': 'Alameda', 'relationship': 'WORKS_FOR'},
{'entity': 'Sam Bankman-Fried', 'related': 'FTX', 'relationship': 'WORKS_FOR'}, {'entity': 'Sam Bankman-Fried', 'related': 'Alameda Research', 'relationship':
'OWNS'}, {'entity': 'Sam Bankman-Fried', 'related': 'Alameda Research', 'relationship': 'FOUNDED'}, {'entity': 'Sam Bankman-Fried', 'related': 'FTX',
'relationship': 'FOUNDED'}]
</item 9>

<item 10>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
“He can’t just walk out if he doesn’t like how it’s going.”



Naftalis spoke at length about what comes next in the trial, and addressed common misconceptions about the prosecution and defense in the latest episode of
TechCrunch’s Chain Reaction podcast, which you can listen to below, or on Apple Podcasts or Spotify.

The SBF trial is one of the biggest cases of its kind within the crypto space. Our resident crypto expert Jacquelyn Melinek has been on the ground at the trial
since it started, and continues to cover the ins and outs of the proceedings. But the trial has been on pause from October 20 to today, so there’s plenty to
catch up on if you need a refresher ahead of it entering its final few phases.

For those in need of a total refresher on the case, her breakdown of how FTX went from the third-largest crypto exchange valued at a peak of $32 billion to
bankruptcy will get you up to speed.
</item 10>
=== END ITEM id=914d4df2487f4d1c ===

=== ITEM id=b9e0c7917d683a61 ===
Gold sentences and facts:
Sentence 0: Time-restricted eating can also improve the quality of sleep, he noted, which can help the brain recover better.
  Fact 0.0: Time-restricted eating can also improve the quality of sleep, according to the speaker.
  Fact 0.1: Better sleep can help the brain recover better, according to the speaker.
Sentence 1: Today, the internet is littered with tips and tricks that promise a sound sleep, but not all of them are effective and none are foolproof.
  Fact 1.0: Today the internet is full of tips and tricks that promise a sound sleep.
  Fact 1.1: Not all of these sleep tips are effective.
  Fact 1.2: None of these sleep tips are foolproof.

Retrieved items:
<item 1>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Joseph Antoun, CEO and chairman of the nutritech company L-Nutra in California, was also involved in the USC study.

"In these studies, mice fed the fasting-mimicking diet showed improved cognitive performance, motor learning and both short- and long-term memory, and promoted
the growth of new neurons in the brain," he told Fox News Digital via email.

"Fasting-mimicking diets have the potential to drastically reduce many aging and disease risk factors by reducing neuroinflammation or superoxide production in
the brain."

Fasting can help the brain function in a myriad of ways, said Antoun.

"Fasting decreases pro-aging factors that are secreted after we eat, which can slow down the aging of the brain," he told Fox News Digital in an emailed
statement.

Time-restricted eating can also improve the quality of sleep, he noted, which can help the brain recover better.
</item 1>

<item 2>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
In her 1973 feminist novel Fear of Flying, author Erica Jong notes that, sometimes, the only way to overcome sleeplessness is by pretending you don’t care about
nodding off. “Then sleep became piqued, like a rejected lover, and crept up to try to seduce you.”

In a world where human sleep patterns have increasingly been plunged into chaos, people are having to turn to ever more creative – and often bizarre – means of
getting their 40 winks in. Today, the internet is littered with tips and tricks that promise a sound sleep, but not all of them are effective and none are
foolproof. (Kiwis, by the way, contain serotonin and antioxidants, which is why it’s thought they might have sleep-promoting benefits.) The result is a society
with a growing problem on its hands: persistent sleeplessness – otherwise known as insomnia.
</item 2>

<item 3>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
The researchers hope that these findings will lead to human trials.

The goal is to obtain funding to start a pilot study of patients next year.

"Accessible interventions like this have great translational potential, as they are available and are usually very well-tolerated," Desplats said.

The goal is to obtain funding to start a pilot study of patients next year.

"This may have special importance for the elderly , as time-restricted eating does not require limitation in calories or change in the diet, but may offer
important benefits ranging from metabolic and sleep regulation to potential cognitive improvement."

TIME-RESTRICTED EATING NO MORE BENEFICIAL THAN CALORIC RESTRICTION IN OBESE PATIENTS, STUDY SAYS

Adopting a new dietary pattern should not be a replacement for medical care , however, as Desplats emphasized the importance of patients' discussions about
options with their clinicians.
</item 3>

<item 4>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
The goal of the restricted eating is to restore the circadian rhythm, the researchers said, which could help counteract the numerous disruptions that
Alzheimer’s patients face when it comes to sleep and schedule-related confusion.

HIDDEN BELLY FAT COULD SIGNAL ALZHEIMER’S DISEASE RISK 15 YEARS BEFORE SYMPTOMS SHOW UP, STUDY FINDS

"Our study emphasizes the power of feeding timing in aligning the circadian clock and its impact on the brain," said senior study author Paula Desplats, PhD,
professor in the Department of Neurosciences at UC San Diego School of Medicine, in a statement to Fox News Digital.

"We were hopeful to see some improvements in pathology, but did not expect such profound effects in reducing plaques and inflammation and improving memory," she
went on.

"The effects of time-restricted feeding on lower amyloid pathology were also measurable in blood using markers found in the clinic, which was another important
finding."

The researchers hope that these findings will lead to human trials.
</item 4>

<item 5>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
One major limitation of the study is that it uses animal models that only represent certain aspects of disease and "do not present comorbidities or other
clinical diseases associated with aging, which are very common in Alzheimer’s patients," noted Desplats.

"Furthermore, the study was done under strictly controlled conditions that did not resemble the variety of people’s habits and environments," she added.

THE 8 BIGGEST ALZHEIMER’S DISEASE MYTHS — AND THE TRUTHS BEHIND THEM

The researchers hope that the public recognizes the importance of the circadian clock in regulating many aspects of health .

"Sustaining healthy sleep and eating habits that are more synchronized with the natural environment are vital in a society that is constantly exposed to food,
light and shift work," said Desplats.

Previous studies have supported this association between fasting and reduced Alzheimer’s risk.
</item 5>

<item 6>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Lavender oil. Eating kiwis. Masturbation. Everyone has their own “hacks” when it comes to falling asleep in 2023; the days of counting sheep are long gone.
Jennifer Aniston’s top tip for battling insomnia is to “sleep with my phone at least five feet away from me”, while Martha Stewart recommends focusing on your
breathing.
</item 6>

<item 7>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“At this point, you’re no longer worried about the traumatic incident; you’re now worried about not being able to sleep.” It’s a self-fulfilling prophecy.

Social media consultant and mother of two, Amy Fieldhouse-Downes, started suffering from insomnia after the birth of her second son in 2019. She believes the
affliction was rooted in anxiety. “If you Google how to go to sleep, I’ve probably tried everything on there,” she says. “Going to bed earlier, going to bed
later, meditations that would get cut off by an advert, lavender products, sleep cream, eye masks, Sominex sleep tablets. And I even used it as an excuse to
have a few glasses of wine.” It wasn’t until she tried sleeping with specialised earplugs that she was finally able to get some rest.
</item 7>

<item 8>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
It’s been proven that what people eat can help prevent or slow Alzheimer’s disease — but what about when they eat?

Participating in intermittent (time-restricted) fasting could lead to a reduced risk of cognitive deterioration, a recent study published in the journal Cell
Metabolism suggests.

Researchers at University of California San Diego School of Medicine adjusted the feeding schedule of certain groups of mice so that they only ate within
six-hour windows each day.

ALZHEIMER’S BLOOD TEST COULD HIT THE MARKET IN EARLY 2024, RESEARCHERS SAY

(This would be equivalent to 14 hours of fasting for humans, the researchers noted.)

Compared to a control group of mice that ate on demand, the fasting mice showed improvements in memory, were less hyperactive in the evenings and had fewer
sleep disruptions .

They also were shown to have less buildup of amyloid proteins in the brain, which is a hallmark of Alzheimer’s disease.
</item 8>

<item 9>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“People who have insomnia are desperate. I treat patients who are suicidal, who have quit their jobs, whose lives are completely on hold because of their
sleeping issues. And they’re vulnerable. My clients have spent hundreds and hundreds of pounds on things that made no difference. If people are being mis-sold
expensive things and they’re being preyed upon, that’s a huge problem.”

Rosie Davies-Smith, a PR consultant, says she has spent more than £600 on a variety of treatments, including a combination of doctor-prescribed and
over-the-counter medications, teas, supplements, weighted blankets, and eye masks. “While supplements, exercise, and sleep hygiene provided some relief, the
true breakthrough came from addressing the root causes of my insomnia through therapy,” she tells me.
</item 9>

<item 10>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
The symptoms of insomnia can last a long time; a longitudinal study conducted in the UK found that 69 per cent of respondents affected by the sleep disorder at
the start of the study still had insomnia 12 months later.

This may go to show how complex treating insomnia can be. Explaining the appeal of quick fixes, Dr Browning says, “In the short term you’re changing something,
so [they] can often [work] like a placebo. If hunger is keeping you awake, eating before bed will help. If your mattress is uncomfortable, sleeping on the floor
will help.” But the problem isn’t just that these methods don’t necessarily work – it’s that they could be used to exploit vulnerable insomniacs.

“I’m a sleep therapist, my doctorate is in insomnia and it’s very, very frustrating,” she says. “People who have insomnia are desperate.
</item 10>
=== END ITEM id=b9e0c7917d683a61 ===

=== ITEM id=1ed4011fb3919f66 ===
Gold sentences and facts:
Sentence 0: Hundreds of women have filed lawsuits against Uber claiming the company hasn’t done enough to prevent instances of sexual assault by drivers.
  Fact 0.0: Hundreds of women have filed lawsuits against Uber.
  Fact 0.1: The women's lawsuits claim Uber hasn't done enough to prevent instances of sexual assault by drivers.
Sentence 1: In the third quarter, Uber generated $394 million in operating income and $219 million worth of net income.
  Fact 1.0: In the third quarter, Uber generated $394 million in operating income.
  Fact 1.1: In the third quarter, Uber generated $219 million worth of net income.

Retrieved items:
<item 1>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Hundreds of women have filed lawsuits against Uber claiming the company hasn’t done enough to prevent instances of sexual assault by drivers. Now a panel of
judges has ruled that more than 80 cases can be consolidated into federal court.

The stakes are high for both parties with implications extending to future Uber riders and drivers. The outcome of the case could result in sweeping changes to
Uber’s platform, which plaintiffs argue will reduce sexual assault and also raise new concerns over privacy.

The upshot? Technology lies at the center of this very human story.

Uber has attempted to address sexual assaults by drivers — which the lawsuits claim Uber has known about since 2014 — through new safety features in its app,
like a 911 button and the ability to share location with a friend. Yet survivors, and their attorneys, say that response has been inadequate, and they’re
calling for better technological solutions like in-vehicle surveillance cameras.
</item 1>

<item 2>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
The number of incidents increased in the categories of nonconsensual kissing, nonconsensual touching and nonconsensual penetration, or rape, between 2017 and
2019.

The total number of rape incidents also rose from 2017 to 2019. And while the number of reported rapes dropped in 2020, the rate has remained the same since
2017. In other words, there has not been an improvement in the rate of rapes on the platform.

Sexual assaults are an ongoing problem for Uber

Uber has been sued numerous times over the past few years by passengers who claim they were sexually assaulted during a ride. Lyft has faced similar lawsuits
and accusations.

Under Uber’s terms of use, class action lawsuits can’t be filed against the company in cases of sexual assault, so each case has to be heard individually. This
has prevented survivors from advocating for themselves collectively.

Judge Charles Breyer in the Northern District of California will preside over pretrial hearings.
</item 2>

<item 3>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
Either way, Cruise said it’s now going to examine “processes, systems, and tools and reflect on how we can better operate in a way that will earn public trust.”

That might be a hefty challenge, especially in California. As the Cruise drama unfolded, opposition against robotaxis grew in cities like Los Angeles. And two
of the biggest groups to oppose robotaxi expansion in California are now formally working together.

Want to reach out with a tip, comment or complaint? Email Kirsten at kirsten.korosec@techcrunch.com or Rebecca at rebecca.techcrunch@gmail.com.

Reminder that you can drop us a note at tips@techcrunch.com. If you prefer to remain anonymous, click here to contact us, which includes SecureDrop
(instructions here) and various encrypted messaging apps.

Micromobbin’

Taiwanese battery swapping giant Gogoro came to play at the Japan Mobility Show 2023, showing off how its scooter batteries can also be used to power a tiny
car.
</item 3>

<item 4>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
After reporting, shares of Uber are up 1.6% in trading, after an up-and-down morning of shares exchange.

Against a backdrop of macroeconomic uncertainty, the former startup darling’s results can be viewed as an indication that its business model has matured and is
now in a stable, profitable and cash-generating position. On the other hand, under-performing demand results from a company as global as Uber could indicate
that consumer spend is coming in lighter than anticipated.

For startups in the transport and on-demand sectors, Uber’s earnings are a regular informational zeitgeist. So let’s work to understand where Uber’s revenue
came from in the third quarter, and how each of its top-line sources converted — or not — to bottom-line results.

Where did the money come from?

In the third quarter, Uber saw total bookings rise from $29.1 billion to $35.3 billion, a gain of around 21%.
</item 4>

<item 5>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
And so if they don’t have drivers, they don’t make money.”

Many drivers install their own dash cams to record trips, usually as backup evidence for insurance claims or to defend themselves against unfair deactivations
from Uber’s platform. Uber is also piloting a new video recording feature for drivers that allows them to record video on their smartphones. But in those cases,
the driver can decide what and when to record, and when to share that data.

Other demands

The survivors in the joined lawsuit also allege that Uber’s “fast and shallow background checks” are substandard and designed to make it as easy as possible for
drivers to sign up quickly. Uber uses third-party companies like Checkr and Appriss to do background checks, which Sergio Avedian, senior contributor at The
Rideshare Guy, says “are at best watered down and not guaranteed of bad apples from falling through the cracks.”

The lawsuit calls on Uber to also include fingerprinting, which would run prospective drivers through FBI databases.
</item 5>

<item 6>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Aside from in-vehicle surveillance and more extensive background checks, the survivors are asking Uber to implement driver training on interactions with
passengers, a zero-tolerance policy for drivers, sexual harassment education and training, and a more adequate system to encourage customer reporting and
monitor customer complaints.

Are Uber’s in-app safety features working?

Abrams argues that Uber’s safety features are insufficient because they’re all app-based and only potentially useful if a rider has access to their own phone.

“The majority of clients I represent and the women I’ve spoken to who have been sexually assaulted have experienced: (A) someone else ordered the ride for them;
(B) their phone is dead or they can’t locate their phone to use the safety shield; or (C) if they could use their phone, they’re incapacitated,” said Abrams.

The lawyer says she has interviewed upward of 5,000 survivors over the course of several years and has seen no difference in the number of attacks before or
after implementation of these in-app features.
</item 6>

<item 7>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
In business segment terms, Uber generated $17.9 billion in ride-hailing bookings (+31% year over year), and $16.1 billion worth of delivery bookings (+18%).
Those key business groups at the company generated $5.1 billion and $2.9 billion in revenue, respectively, during the September quarter.

There’s nuance to the revenue figures that we need to consider as the rule-changes in question do affect startups that operate in related categories. On the
ride-hailing front, Uber told its investors that its Q3 2023 revenue was “negatively impacted by business model changes in some countries that classified
certain sales and marketing costs as contra revenue by $161 million.” That was not the only legal change that led to Uber’s results changing shape. Under its
delivery business results, the company added that its revenue result was “negatively impacted by business model changes that classified certain sales and
marketing costs as contra revenue by $360 million.”

The combined impact of those two items was 8 percentage points of growth.
</item 7>

<item 8>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Uber hasn’t released safety data for 2021 and 2022, and won’t say when it plans to do so. Older data backs up some of Abrams’ claims.

Uber has published two safety reports — one containing data from 2017 to 2018 and another with data from 2019 to 2020. The ride-hail company claims that the
rate of sexual assault reported on the app decreased 38% between its first and second reports.

While the rate decrease is positive, not all the data delivers the same message. Over that same time period, the total number of sexual assault reports across
five categories fell from 5,981 to 3,824. That drop could be explained by the decrease of trips from 1.4 billion in 2019 to 650 million in 2020.

And while the total number of incidents as a percentage of total number of rides does decrease through the years, the number of incidents in certain categories
actually increases.
</item 8>

<item 9>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Uber reported third-quarter earnings on Tuesday that show a profitable ride-hail and delivery company that’s chugging along in spite of slowing growth in some
sectors.

The company reported revenue of $9.3 billion, an 11% increase year over year. Investors had expected Uber to report revenues of around $9.5 billion (FactSet,
Refinitiv), meaning that despite the company’s growth, it fell short of estimates. Turning to profitability, Uber reported net income of $221 million in the
third quarter, or 10 cents per share, compared with a net loss of $1.2 billion, or 61 cents per share, in the same quarter last year. Again, the company fell
short of expectations that it would generate 12 cents in per-share profit.

Looking ahead, Uber anticipates gross bookings of $36.5 billion to $37.5 billion, up just 6% at the top-end from its Q3 result.
</item 9>

<item 10>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Of course, adjusted EBITDA is to profit as hidden heels in men’s shoes are to height, so we’ll want some harder figures as well. In the third quarter, Uber
generated $394 million in operating income and $219 million worth of net income. Mix in $966 million worth of positive operating cash flow, and Uber looks very
healthy, given that all of the figures in this paragraph were improvements on its year-ago results.

But while food delivery and scooting humans about town were profitable for Uber in the quarter, the final major portion of its business had a more lackluster
quarter.

Freight falls

One area that continues to drag on Uber is its freight business.

While Uber’s ride-hailing and delivery business saw an uptick in gross bookings in the third quarter, Uber Freight experienced a 27% drop year-over-year.
Revenue, as a result, had a similar fall.

The business unit reported revenue of $1.3 billion in the third quarter, a 27% drop from the same period last year.
</item 10>
=== END ITEM id=1ed4011fb3919f66 ===
