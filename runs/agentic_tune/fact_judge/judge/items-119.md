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

=== ITEM id=d9fa0aa3475aa797 ===
Gold sentences and facts:
Sentence 0: Shares of Rogers extended losses on news of the lawsuit, closing down 2.2% to C$53.07 in Toronto.
  Fact 0.0: Shares of Rogers extended losses on news of the lawsuit.
  Fact 0.1: Rogers shares closed down 2.2% to C$53.07 in Toronto.
Sentence 1: Amazon tumbled 4 per cent after the Federal Trade Commission and 17 state attorneys general filed an antitrust lawsuit against it.
  Fact 1.0: The Federal Trade Commission filed an antitrust lawsuit against Amazon.
  Fact 1.1: 17 state attorneys general filed an antitrust lawsuit against Amazon.
  Fact 1.2: Amazon tumbled 4 per cent after the lawsuit.

Retrieved items:
<item 1>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“Just like any store owner who wouldn’t want to promote a bad deal to their customers, we don’t highlight or promote offers that are not competitively priced.”
Zapolsky argued the FTC’s lawsuit could force it to stop highlighting low prices, “a perverse result that would be directly opposed to the goals of antitrust
law.”

“Long overdue” lawsuit

On Amazon’s own forum for merchants, called Seller Central, several users cheered on the FTC and said they hoped it would result in changes to the company’s
business practices. Amazon’s tense relationship with merchants has been well-chronicled over the years, with sellers expressing a range of grievances over
issues like rising fees, an arcane suspensions process, and heightened competition on the marketplace from all sides, including the e-commerce giant. “I think
it’s great, Amazon deserves it,” one person commented, adding, “More should be coming on the way.”

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Watchara Phomicinda
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Fred Ruckel
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> MediaNews Group

Entities: Zapolsky, Seller Central, FTC antitrust case, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, FTC, antitrust law, CNBC, Amazon
</item 1>

<item 2>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Hart said Amazon sales comprise about 90% of his business, meaning any hit those sales take on Amazon has a considerable impact. He recalled “24 anxious hours”
in September 2022 when a third-party seller of his popular construction toy Brain Flakes listed the toy for a lower price on Target than it was offered on
Amazon.

Molson Hart, CEO of Viahart, an educational toy company that sells on Amazon. Courtesy: Molson Hart

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Fred Ruckel
- Amazon -ACCUSED_OF-> FTC antitrust case
- Zapolsky -SAID-> FTC antitrust case
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale

Entities: FTC antitrust case, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, Viahart, Molson Hart, Amazon, 24 anxious hours, Brain Flakes,
FTC, CNBC, Target
</item 2>

<item 3>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
A worker sorts out parcels in the outbound dock at the Amazon fulfillment center in Eastvale, California, on Aug. 31, 2021. Watchara Phomicinda | MediaNews
Group | The Riverside Press-Enterprise via Getty Images

It was late in the day on Oct. 27, 2021, when Fred Ruckel received the dreaded automated email from Amazon . Amazon’s software had detected that Ruckel’s
popular cat toy, called the Ripple Rug, was being sold somewhere else for a cheaper price. His product would no longer be shown in Amazon’s all-important buy
box, an area of the listing where shoppers click “Add to Cart.” Ruckel is the sole seller of the Ripple Rug on Amazon, so the move all but ensured his product
would disappear from the website, costing him thousands of dollars per day.

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Snuggly Cat
- Snuggly Cat -MADE-> Ripple Rug
- Zapolsky -SAID-> FTC antitrust case
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Getty Images

Entities: Fred Ruckel, Eastvale, Watchara Phomicinda, MediaNews Group, The Riverside Press-Enterprise, Amazon fulfillment center, Ripple Rug, FTC antitrust
case, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, Getty Images, California, FTC, CNBC, Amazon
</item 3>

<item 4>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Fred Ruckel

Nearly three years later, Ruckel’s experience hits at the core of a sweeping antitrust lawsuit filed last week by the Federal Trade Commission against Amazon.
The agency accused Amazon of wielding its monopoly power to squeeze merchants and thwart rivals. For consumers, that’s led to artificially inflated prices and a
degraded shopping experience, the agency alleges. In the 172-page suit, the FTC said Amazon relies on an “anti-discounting strategy” and a “massive web-crawling
apparatus that constantly tracks online prices” to stifle competition. The agency said Amazon punishes third-party sellers who offer cheaper products elsewhere
by threatening to disqualify them from appearing in the buy box if it detects a lower price. Losing the buy box is an “existential threat” to sellers’
businesses, the complaint alleges. The end result of these tactics, the FTC argues, is elevated prices across the web.

Graph facts:
- 5 things we learned so far about the Google antitrust case -MENTIONS-> monopoly power
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale
- Federal Trade Commission -PARTICIPATED_IN-> antitrust lawsuit against Amazon
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Watchara Phomicinda

Entities: Federal Trade Commission, Fred Ruckel, sweeping antitrust lawsuit, monopoly power, Amazon sellers sound off on the FTC's 'long-overdue' antitrust
case, CNBC, Amazon
</item 4>

<item 5>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers. Soon after, then-CEO Joe Natale was sacked and
replaced by Staffieri, the former chief financial officer. Rogers-Hixon and Martha Rogers had opposed the CEO change.

In early 2022, family members agreed to temporarily shelve their disagreements, according to this week’s court filing, to avoid negative publicity as the
company sought regulators’ approval for the takeover of Shaw Communications Inc. The C$20 billion transaction was the company’s biggest deal ever and among the
largest corporate transactions in Canadian history.

Negotiations Collapse

As part of that truce, Rogers-Hixon and Martha Rogers agreed not to participate in board discussions about the Shaw deal or receive materials on it, because of
an argument about Rogers-Hixon’s personal lawyer, whose firm also does work for Telus Corp., a competitor to Rogers and Shaw.

Graph facts:
- Edward Rogers -WORKS_FOR-> Rogers Communications Inc.
- Edward Rogers -WORKS_FOR-> family trust
- Edward Rogers -PARTICIPATED_IN-> board battle
- Edward Rogers -ACCUSED_OF-> personal vendetta
- Martha Rogers -PARTICIPATED_IN-> court petition filing

Entities: Edward Rogers, Martha Rogers, Rogers Communications, Joe Natale, Staffieri, Shaw Communications Inc., Telus Corp., takeover of Shaw Communications
Inc., CEO change, Rogers-Hixon, Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’, Fortune
</item 5>

<item 6>
Graph query row:
article: Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’
publisher: Fortune
passage: Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
Two members of the billionaire family that controls Rogers Communications Inc. are suing the company, saying they’ve been frozen out of board meetings and
blocked from getting information, reigniting a bitter power struggle within one of Canada’s wealthiest clans.

Melinda Rogers-Hixon says she’s the target of a “personal vendetta” by her older brother, Rogers Chair Edward Rogers, according to a letter contained in the
court filing. The Rogers siblings were close to an agreement that would have resolved their longstanding differences, but Edward Rogers reneged on it, according
to the Sept. 20 missive by Rogers-Hixon, which was sent to Chief Executive Officer Tony Staffieri.

The court petition was filed by Rogers-Hixon and her sister, Martha Rogers, and it argues the company has been “oppressive” to them.

The documents show that the fractures inside the prominent Toronto family have never healed.
</item 6>

<item 7>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The Shaw takeover finally closed on April 3 of this year, and the Rogers siblings engaged in talks “intended to resolve all issues between us,” Rogers-Hixon
stated in an affidavit. “I was hopeful all issues could be resolved without negative publicity.” But those negotiations collapsed last month, she said.

It isn’t clear from the court filings whether the discussions included the possibility that any Rogers family members would sell their stake in Rogers
Communications, leave their roles at the company, or exit the family trust that controls almost all of the voting shares of the public company.

“This matter should be resolved privately,” Rogers Communications spokesperson Sarah Schmidt said by email. “We’ve demonstrated that we will not be distracted
by these actions — we have significant momentum in the market, our merger is tracking ahead of plan, and we remain squarely focused on doing what’s right for
our customers and stakeholders.”

Graph facts:
- Rogers Communications -PARTICIPATED_IN-> takeover of Shaw Communications Inc.
- Joe Natale -WORKS_FOR-> Rogers Communications
- Staffieri -WORKS_FOR-> Rogers Communications
- Schmidt -SAID-> Rogers Communications
- Rogers Communications -OWNS-> Toronto Blue Jays

Entities: Rogers Communications, Sarah Schmidt, Rogers family trust, Shaw, Shaw takeover, merger, Rogers-Hixon, Power struggle in one of Canada’s wealthiest
clans has sister accusing chairman brother of ‘vendetta’, Fortune
</item 7>

<item 8>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Shopify, for example, gives sellers much more control over how they communicate with customers, Needham said, adding that although Amazon still controls the
communication process, at least there is one. "I wish it was a clear-cut case,” Needham said. “I have a vested interest in the marketplace doing really well, as
a seller and as a service provider. And... this case, it doesn’t make the marketplace better for sellers."

Concerns over Amazon pricing policies, fees

Many sellers have zeroed in on Amazon’s pricing policies and rising fees as rightful areas of concern in the FTC's lawsuit. Molson Hart, whose company Viahart
sells toys on Amazon, has been a longtime critic of Amazon’s pricing policies. Hart complained of how Amazon’s seller fees impact pricing in a 2019 Medium post
and later that year testified about his experience before a House committee.

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Fred Ruckel
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale
- Needham -SAID-> Fulfillment by Amazon
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Watchara Phomicinda

Entities: Needham, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, Viahart, House committee, FTC lawsuit, Amazon's pricing policies, Amazon
seller fees, Testimony before a House committee, Toys, Molson Hart, Shopify, FTC, CNBC, Amazon, Medium
</item 8>

<item 9>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The documents show that the fractures inside the prominent Toronto family have never healed. Rogers Communications is the country’s largest wireless provider
and the owner of a cable television empire, the Toronto Blue Jays baseball team and other media and sports assets. The family’s controlling stake in the public
company is worth close to C$8 billion ($5.9 billion), according to calculations based on company filings for its 2023 annual meeting.

Shares of Rogers extended losses on news of the lawsuit, closing down 2.2% to C$53.07 in Toronto. They’ve dropped 16% this year.

The children of late founder Ted Rogers have frequently fought one another — sometimes driving down Rogers’ stock price as investors grew worried about the
company’s governance.

The bad blood spilled into full view in October 2021, when Edward Rogers publicly battled his sisters and his mother for control of the board and won in court.
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Toronto Blue Jays
- Amidst Managerial Shake-Ups, These 5 MLB Teams Are Looking for a New Leader: Who Are the Top Candidates? -MENTIONS-> Toronto Blue Jays
- Don Mattingly -WORKS_FOR-> Toronto Blue Jays
- Toronto Blue Jays -PARTICIPATED_IN-> Wild Card sweep
- Edward Rogers -WORKS_FOR-> Rogers Communications Inc.

Entities: Toronto Blue Jays, Edward Rogers, Ted Rogers, board battle, company governance, vendetta, Rogers Communications, Power struggle in one of Canada’s
wealthiest clans has sister accusing chairman brother of ‘vendetta’, Fortune, Toronto, lawsuit, Canada
</item 9>

<item 10>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
Two members of the billionaire family that controls Rogers Communications Inc. are suing the company, saying they’ve been frozen out of board meetings and
blocked from getting information, reigniting a bitter power struggle within one of Canada’s wealthiest clans.

Melinda Rogers-Hixon says she’s the target of a “personal vendetta” by her older brother, Rogers Chair Edward Rogers, according to a letter contained in the
court filing. The Rogers siblings were close to an agreement that would have resolved their longstanding differences, but Edward Rogers reneged on it, according
to the Sept. 20 missive by Rogers-Hixon, which was sent to Chief Executive Officer Tony Staffieri.

The court petition was filed by Rogers-Hixon and her sister, Martha Rogers, and it argues the company has been “oppressive” to them.

The documents show that the fractures inside the prominent Toronto family have never healed.

Graph facts:
- Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ -MENTIONS-> Toronto Blue Jays
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune
- Edward Rogers -WORKS_FOR-> family trust
- Edward Rogers -PARTICIPATED_IN-> board battle
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune

Entities: Rogers Communications Inc., Melinda Rogers-Hixon, Edward Rogers, Martha Rogers, Tony Staffieri, personal vendetta, oppression, court petition filing,
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’, Fortune, Toronto, Canada
</item 10>
=== END ITEM id=d9fa0aa3475aa797 ===

=== ITEM id=449036c4c71e5d72 ===
Gold sentences and facts:
Sentence 0: From Netflix: Jess and Bill Walker are doing their best to keep their family connected as their children grow older, more independent, and more distant.
  Fact 0.0: The item is from Netflix.
  Fact 0.1: Jess and Bill Walker are doing their best to keep their family connected.
  Fact 0.2: The Walkers' children are growing older.
  Fact 0.3: The Walkers' children are growing more independent.
  Fact 0.4: The Walkers' children are growing more distant.
Sentence 1: Speaking of the Christmas season, McLean — born in New Rochelle, New York — said that "the holidays are a time for families to come together.
  Fact 1.0: McLean was born in New Rochelle, New York.
  Fact 1.1: McLean spoke about the Christmas season.
  Fact 1.2: McLean said the holidays are a time for families to come together.

Retrieved items:
<item 1>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“I’d recommend pulling your partner aside and saying: ‘How are we doing?’ Like a temperature check,” she added. “Those are points where the two of you can come
and remind each other: ‘We are a team in this.’ Or, the two of you could go on a walk and separate from the family, just for a little bit of time to reset
within the relationship.”

During the holiday season, your partner may be seeing how you interact with your family for the first time. But that’s not necessarily a bad thing, according to
the relationship experts. “I think introducing a significant other to family is a really vulnerable act, and that vulnerability is going to be what allows for
greater intimacy,” Lurie said. “When your significant other sees you with your family, they may see another side of you that they may not have had access to
before.”

(Getty Images)

On the other hand, your partner could learn that your relationship with your parents and extended family members is complex.

Graph facts:
- Country music couples Tim McGraw, Faith Hill and Garth Brooks, Trisha Yearwood share tips to lasting marriages -MENTIONS-> Family
- A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation -MENTIONS-> Family
- Who Are Mikaela Shiffrin’s Parents? All You Need to Know About Skiing Legend’s Family -MENTIONS-> Family
- The best comedies you’ll find on streaming right now -MENTIONS-> Family
- Jada Pinkett Smith -SAID-> Family

Entities: Family, Should you introduce your partner to your family during the holidays? Relationship experts weigh in, Intimacy, Lurie, Getty Images,
relationships, The Independent
</item 1>

<item 2>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
The Santa bribe turns the focus of Christmas into "selfish gain on the part of your children," Keeley added.

"Love your kids enough to learn how to manage their behavior appropriately and teach them self-discipline."

"It’s also lazy parenting, which never pays off in the long run," she said. "Being a mom is a professional career and deserves to be treated accordingly."

She added, "Love your kids enough to learn how to manage their behavior appropriately and teach them self-discipline."

3. Becoming a ‘holiday martyr’

It’s important to keep the holidays well-rounded, Keeley said.

5 HOLIDAY DECORATING MISTAKES THAT COULD REDUCE YOUR CHEER THIS YEAR

She called out "martyr moms" who are at their "peak performance" during Christmastime, putting everyone else before themselves.

"They wear themselves out making sure everyone else has the perfect Christmas, but never put themselves on the receiving end," she said.

Graph facts:
- Keeley -SAID-> child brain
- Keeley -SAID-> grinchy griping
- Keeley -SAID-> egocentric information processing
- Keeley -SAID-> mom brain
- Keeley -SAID-> give list

Entities: Santa bribe, self-discipline, holiday martyrdom, Keeley, The 5 biggest mistakes parents make at Christmastime, according to a parenting expert,
Christmas, FOX News - Lifestyle
</item 2>

<item 3>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
That isn’t necessarily a cause for concern, as long as you inform your significant other about the interworkings of your family dynamics.

“I think that we tend to be our best relational self when we’ve really done the work around understanding all of our past relationships,” Zar explained. “You
can tell your partner: ‘I don’t have a close relationship with my brother, and here’s why. Here are the intentional choices that I’ve made around it.’ Or you
can say: ‘Here’s my chosen family. These are the people who are important to me.’ Your significant other will see that you’ve been introspective and have
takeaways around how you’ve navigated your family’s origin.”

As long as you and your partner have discussed your relationship expectations - both before and during the holidays – introducing them to your family may deepen
your relationship and could be a step in the right direction.

Graph facts:
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -PUBLISHED_BY-> The Independent - Life and Style
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Family
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Rachel Zar
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Spark Chicago Therapy
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Hanukkah

Entities: Should you introduce your partner to your family during the holidays? Relationship experts weigh in, Zar, family dynamics, past relationships,
relationship expectations, chosen family, The Independent, holidays
</item 3>

<item 4>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
Pressures are high for parents and caregivers around the holidays.

That means there’s room for error, according to parenting expert Hannah Keeley, who offered up some of the biggest mistakes parents tend to make at
Christmastime.

In an email she sent to Fox News Digital, Virginia-based Keeley identified the top five holiday blunders that parents often make.

HOME FOR THE HOLIDAYS? HERE'S HOW TO AVOID STRESS AROUND FAMILY MEMBERS, ACCORDING TO EXPERTS

Here are the mistakes Keeley thinks parents should try to avoid.

Diving right in …

1. Creating unrealistic expectations

Parents who make a "must-do" Christmas list that’s a "mile long" at the "first sniff of peppermint mochas" are perhaps setting themselves up for failure — and
Keeley said she empathized with this instinct.

Doing the most could lead to a season of disappointment, she warned.

CHRISTMAS TREE SHOPPING?

Graph facts:
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -MENTIONS-> Virginia
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> Virginia
- The obsessive tormenter who made professors’ lives miserable -MENTIONS-> Virginia
- Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars
Sportsbook, DraftKings and FanDuel -MENTIONS-> Virginia
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Virginia

Entities: Virginia, Hannah Keeley, Christmastime, unrealistic expectations, holiday blunders, The 5 biggest mistakes parents make at Christmastime, according to
a parenting expert, Fox News, Fox News Digital, parenting
</item 4>

<item 5>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“Sometimes it can be helpful for them to know about the dynamics that they’re stepping into and how they can best support you. Frankly, they should be aware if
there is tension or if there is stress within your family.”

(Getty Images)

Communication before the holidays goes both ways, and there are things that your family should know about your significant other too. “If there’s something
about your partner that could be surprising to your family, or that’s a notable difference from who someone you’ve brought home before, tell your relatives
that,” Zar said. “Let’s set your family and partner up for success.”

Before bringing your significant other home for the holidays, you may want to express to your family how you wish for them to feel included - such as
incorporating your partner’s family traditions into your own celebration. However, Zar advised against oversharing information before Christmas. You still want
your family and partner to connect in person, rather than hearing about each other through the grapevine.

Graph facts:
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -PUBLISHED_BY-> The Independent - Life and Style
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Family
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Rachel Zar
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Spark Chicago Therapy
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Hanukkah

Entities: Should you introduce your partner to your family during the holidays? Relationship experts weigh in, Zar, Getty Images, The Independent
</item 5>

<item 6>
Graph query row:
title: None
publisher: Fox News
subjects: ['Santa Claus', 'Hannah Keeley', 'Christmastime', 'unrealistic expectations', 'holiday blunders', 'Hallmark', 'Grinch movie', 'Santa bribe',
'self-discipline', 'holiday martyrdom', 'holiday stress', 'self-worth', 'Grinch', 'cut-out sugar cookies', 'egocentric information processing', 'Keeley', 'mom
brain', 'child brain', 'grinchy griping', 'give list', 'wish list', 'parents', 'children', 'Christmas', 'gifts', 'credit card', 'imagination', 'creative
ability', 'receiving and giving', 'parenting expert', 'FOX News app', 'Earth', 'Fox News Digital', 'parenting']
passages: ['The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)\nPressures
are high for parents and caregivers around the holidays.\n\nThat means there’s room for error, according to parenting expert Hannah Keeley, who offered up some
of the biggest mistakes parents tend to make at Christmastime.\n\nIn an email she sent to Fox News Digital, Virginia-based Keeley identified the top five
holiday blunders that parents often make.\n\nHOME FOR THE HOLIDAYS? HERE\'S HOW TO AVOID STRESS AROUND FAMILY MEMBERS, ACCORDING TO EXPERTS\n\nHere are the
mistakes Keeley thinks parents should try to avoid.\n\nDiving right in …\n\n1. Creating unrealistic expectations\n\nParents who make a "must-do" Christmas list
that’s a "mile long" at the "first sniff of peppermint mochas" are perhaps setting themselves up for failure — and Keeley said she empathized with this
instinct.\n\nDoing the most could lead to a season of disappointment, she warned.\n\nCHRISTMAS TREE SHOPPING?', 'The 5 biggest mistakes parents make at
Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)\nCHRISTMAS TREE SHOPPING? HERE\'S HOW TO SELECT THE PERFECT ONE
FOR YOU\n\n"Aside from the time I asked my husband to turn our bathroom into a Hallmark movie and he actually pulled it off, unrealistic expectations around
Christmas usually just set up moms for inevitable disappointment," Keeley said.\n\n"Beautiful family Christmas memories can be made without the drive-in
‘Grinch’ movie, the tacky light tour and the neighborhood Christmas cookie bake-off all in the same weekend."\n\nInstead, parents should "take it easy and leave
plenty of room to relax," said Keeley.\n\n2. Using Christmas as a bribe\n\nWhile it’s easy for parents to use a message of "be good or else" as a parenting
tactic before Santa Claus comes down the chimney, Keeley said this approach lowers parenting standards.\n\n"I refer to this tactic as ‘stooping to Santa,’" she
said. "Don’t lower your parenting standards to bribery, especially around a season and a character that is supposed to symbolize peace and goodwill on Earth."',
'The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)\nThe Santa bribe turns
the focus of Christmas into "selfish gain on the part of your children," Keeley added.\n\n"Love your kids enough to learn how to manage their behavior
appropriately and teach them self-discipline."\n\n"It’s also lazy parenting, which never pays off in the long run," she said. "Being a mom is a professional
career and deserves to be treated accordingly."\n\nShe added, "Love your kids enough to learn how to manage their behavior appropriately and teach them
self-discipline."\n\n3. Becoming a ‘holiday martyr’\n\nIt’s important to keep the holidays well-rounded, Keeley said.\n\n5 HOLIDAY DECORATING MISTAKES THAT
COULD REDUCE YOUR CHEER THIS YEAR\n\nShe called out "martyr moms" who are at their "peak performance" during Christmastime, putting everyone else before
themselves.\n\n"They wear themselves out making sure everyone else has the perfect Christmas, but never put themselves on the receiving end," she said.', 'The 5
biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)\nThese parents, said Keeley,
"subconsciously compensate for a lack of self-worth and validate their role as a [parent] by creating too much busyness in their life."\n\nShe said that "all
that results from a mom who runs herself into the ground is just that: a mom who runs herself into the ground."\n\nDELICIOUS CUT-OUT SUGAR COOKIES FOR
CHRISTMAS: TRY THE SIMPLE RECIPE\n\nShe added, "If this is you, take a breath, put away the to-do list and ask yourself, ‘What would I have to believe in order
to have a relaxed and joyful Christmas?’ Maybe we can use that as a guide to sit back and relax."\n\n4. Griping like the Grinch\n\nParents should be sure not to
be "grinchy" around their kids amid holiday stress, Keeley said.\n\nChildren up to the age of 9 process information egocentrically, she said — which means if
there’s a problem, they see themselves as the cause.', 'The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News -
Lifestyle, 2023-12-09T19:46:55+00:00)\nWith this in mind, parents should avoid "processing their frustration around the holidays audibly around the children,"
she said.\n\n"The ‘mom brain’ is wired to process problems rapidly and move on to solutions," she said. "The ‘child brain,’ however, is wired to process a
problem internally and take on the responsibility."\n\nSaid Keeley, "Knowing this, make sure you don’t do any ‘grinchy griping’ out loud to the children. You
may see it as temporary venting, but they see it as, ‘Mom is upset and I’m probably the reason.’"\n\nCLICK HERE TO SIGN UP FOR OUR LIFESTYLE
NEWSLETTER\n\nInstead, parents should stay calm, find a reason to be grateful and vocalize it, even when frustrations accumulate.\n\n5. Neglecting the ‘give
list’\n\nWhile building a wish list is fun for kids, Keeley emphasized the moral importance of the "give list."', 'The 5 biggest mistakes parents make at
Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)\n"Many parents make the mistake of teaching their children that
Christmas is a time to make a ‘wish list’ and get what they want," she said.\n\n"Many parents make the mistake of teaching their children that Christmas is a
time to make a ‘wish list’ and get what they want."\n\n"This is loads of fun, but can often create a big case of \'the gimmes’ in your children."\n\nShe added,
"Life is balance. It’s about receiving and giving."\n\nAs parents guide their kids to create a wish list, Keeley encouraged them to spend equal time creating a
list of gifts to give to family and friends.\n\n"Collaborate with them on what they want to give to the people they love, and encourage them to use their
imagination and creative ability, not just your credit card," she said.', 'The 5 biggest mistakes parents make at Christmastime, according to a parenting expert
(FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)\nCLICK HERE TO GET THE FOX NEWS APP\n\n"Even if it\'s a pine cone covered in peanut butter and bird seed for
loved ones to use as a bird feeder, the gift came from the heart and a beautiful lesson was learned — that’s the best Christmas gift of all," she said.']
</item 6>

<item 7>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
“I think anytime that emotions are heightened, with the holidays certainly being one of those times, our expectations are really high,” she said. “We’re
expecting it to be this joyful, happy, perfect occasion. So, we sometimes forget that the holidays come with a lot of stress.”

Despite tensions between relatives at the dinner table, that doesn’t mean your relationship shouldn’t be celebrated. Saba Lurie is a licensed marriage and
family therapist at Take Root Thereapy in California. She explained to The Independent that while there isn’t a right or wrong time to introduce your partner to
your family, the intentions of such an introduction must be clear. Not only should you have already discussed where your relationship is headed, but you should
also tell your partner everything they need to know about your family.

“I think you should tell the person why you’re bringing them home for the holidays, so you’re on the same page about the intentions and what’s motivating the
decision to include,” she said.

Graph facts:
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -PUBLISHED_BY-> The Independent - Life and Style
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Family
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Rachel Zar
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Spark Chicago Therapy
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Hanukkah

Entities: Should you introduce your partner to your family during the holidays? Relationship experts weigh in, Saba Lurie, Take Root Thereapy, California, The
Independent
</item 7>

<item 8>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
These parents, said Keeley, "subconsciously compensate for a lack of self-worth and validate their role as a [parent] by creating too much busyness in their
life."

She said that "all that results from a mom who runs herself into the ground is just that: a mom who runs herself into the ground."

DELICIOUS CUT-OUT SUGAR COOKIES FOR CHRISTMAS: TRY THE SIMPLE RECIPE

She added, "If this is you, take a breath, put away the to-do list and ask yourself, ‘What would I have to believe in order to have a relaxed and joyful
Christmas?’ Maybe we can use that as a guide to sit back and relax."

4. Griping like the Grinch

Parents should be sure not to be "grinchy" around their kids amid holiday stress, Keeley said.

Children up to the age of 9 process information egocentrically, she said — which means if there’s a problem, they see themselves as the cause.

Graph facts:
- Keeley -SAID-> holiday martyrdom
- Keeley -SAID-> mom brain
- Keeley -SAID-> Santa bribe
- Keeley -SAID-> self-discipline
- Keeley -SAID-> child brain

Entities: holiday stress, self-worth, Grinch, cut-out sugar cookies, egocentric information processing, Keeley, The 5 biggest mistakes parents make at
Christmastime, according to a parenting expert, Christmas, FOX News - Lifestyle
</item 8>

<item 9>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
In a recent poll from the American Psychological Association, 41 per cent of US adults said their stress levels increased from November to January when compared
to other months of the year. Nearly 50 per cent of Americans also described their stress level during the season as “moderate”.

Everyone has different concerns leading up to Christmas, Hanukkah, or New Years - from family drama at the dinner table to a small spats with your mother about
decorations. With what can be stressful events, it only begs the question: do you really want to bring your partner into the mix? While it’s important to
eventually introduce your loved ones to your significant other, relationship experts believe there are some discussions you should have beforehand, especially
during the holidays.

Speaking to The Independent, Rachel Zar - a licensed marriage and family therapist at Spark Chicago Therapy - acknowledged that while every family is different,
there’s a variety of problems that could arise during the holiday season - mainly because we put too much pressure on having a joyful holiday.

Graph facts:
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Take Root Thereapy
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Family
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -PUBLISHED_BY-> The Independent - Life and Style
- Should you introduce your partner to your family during the holidays? Relationship experts weigh in -MENTIONS-> Saba Lurie
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -MENTIONS-> American
Psychological Association

Entities: Should you introduce your partner to your family during the holidays? Relationship experts weigh in, Rachel Zar, Spark Chicago Therapy, Hanukkah, New
Years, family problems, American Psychological Association, Christmas, The Independent, stress
</item 9>

<item 10>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
With this in mind, parents should avoid "processing their frustration around the holidays audibly around the children," she said.

"The ‘mom brain’ is wired to process problems rapidly and move on to solutions," she said. "The ‘child brain,’ however, is wired to process a problem internally
and take on the responsibility."

Said Keeley, "Knowing this, make sure you don’t do any ‘grinchy griping’ out loud to the children. You may see it as temporary venting, but they see it as, ‘Mom
is upset and I’m probably the reason.’"

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Instead, parents should stay calm, find a reason to be grateful and vocalize it, even when frustrations accumulate.

5. Neglecting the ‘give list’

While building a wish list is fun for kids, Keeley emphasized the moral importance of the "give list."

Graph facts:
- Keeley -SAID-> Santa bribe
- Keeley -SAID-> self-discipline
- Keeley -SAID-> holiday martyrdom
- Keeley -SAID-> egocentric information processing
- Keeley -SAID-> Christmas

Entities: Keeley, mom brain, child brain, grinchy griping, give list, wish list, The 5 biggest mistakes parents make at Christmastime, according to a parenting
expert, FOX News - Lifestyle
</item 10>
=== END ITEM id=449036c4c71e5d72 ===

=== ITEM id=30b86cd61a8ddaf2 ===
Gold sentences and facts:
Sentence 0: Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front, Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.
  Fact 0.0: Caroline Ellison was Bankman-Fried's on-again off-again girlfriend.
  Fact 0.1: Caroline Ellison was the CEO of Alameda.
  Fact 0.2: The government claimed Bankman-Fried used Caroline Ellison as a front.
  Fact 0.3: The government claimed Bankman-Fried had 'secret access' to customer money, both cash and crypto.
Sentence 1: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 1.0: The second week of the trial's standout testimony came from Caroline Ellison.
  Fact 1.1: Caroline Ellison is Alameda Research's former CEO.
  Fact 1.2: Caroline Ellison claimed she took $14 billion from customers.
  Fact 1.3: Caroline Ellison claimed she took the $14 billion to repay debts to lenders.
  Fact 1.4: Caroline Ellison claimed she used customers as a line of credit.
  Fact 1.5: Caroline Ellison claimed she acted under the instruction of SBF.

Retrieved items:
<item 1>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda got its war chest — 60 percent to 70 percent of the initial supply — for free, while seed investors got FTT at 10 cents a coin, and FTT first listed at
$1 a coin. Bankman-Fried felt that $1 per coin was psychologically important, Ellison said, and he directed her to buy up FTT using Alameda if its price fell
below a dollar.

FTT was one of several “Sam coins,” a nickname for tokens that Bankman-Fried was heavily involved in and owned a lot of, either personally or through Alameda.
Those coins were almost certainly worth less than the value displayed on the balance sheet because trying to sell them all at once would crater the prices.
Bankman-Fried directed her to put those coins on the balance sheets Alameda showed to lenders, even though she felt it was “somewhat misleading.”

Alameda was also getting loans from outside lenders, such as Genesis, because when FTX started, there weren’t a lot of customer funds to borrow, Ellison
testified.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Sam Bankman-Fried was a terrible boyfriend, FTT, Sam coins, Ellison, FTX, Sam Bankman-Fried, Genesis, The Verge
</item 1>

<item 2>
Graph query row:
article: SBF Trial: The latest updates from the FTX collapse’s courtroom drama
publisher: TechCrunch
connection: SBF Trial
passage: SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 2>

<item 3>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That gave Ellison an unusual view of his character. “He was very ambitious,” she said. Besides telling her about his presidential chances, he also told her that
if there was a coin flip where tails destroyed the world and heads made the world twice as good, he’d flip the coin. He called this being “risk-neutral,” which
seems like a fancy way of saying he was a gambling addict.

She was named co-CEO of Alameda with Sam Trabucco in 2021, while she and Bankman-Fried were broken up, and CEO in 2022. The goal, Bankman-Fried told her, was to
“optically” separate Alameda Research and FTX. “The whole time we were dating, he was my boss at work,” she said. They broke up because she wanted more from the
relationship; Bankman-Fried was distant and not paying enough attention to her.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda Research
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda Research
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda Research
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda Research
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda Research

Entities: Alameda Research, Sam Trabucco, Sam Bankman-Fried was a terrible boyfriend, Ellison, FTX, Sam Bankman-Fried, The Verge
</item 3>

<item 4>
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
</item 4>

<item 5>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
(Fraud, conspiracy to commit fraud, and money laundering, in case you were wondering.) “We ultimately took about $14 billion, some of which we were not able to
pay back.” She tilted her head down to answer the questions, then lifted her head when she’d finished her answer.

Bankman-Fried called this being “risk-neutral,” which seems like a fancy way of saying he was a gambling addict

In Ellison’s telling, Alameda was troubled from her earliest time there in 2018. “Shortly after I started, I learned the company was in worse shape than I
realized,” Ellison said. Alameda had initially been funded with loans “from acquaintances,” she said, and those loans were recalled a few weeks after she
arrived. (There was a staff revolt within Alameda Research over lost millions and general financial chaos, according to Michael Lewis’ Going Infinite.) Ellison
asked Bankman-Fried why he hadn’t shared the company’s shaky circumstances in the job offer. “He hadn’t known how to tell me,” she said.

Graph facts:
- Michael Lewis -PARTICIPATED_IN-> Michael Lewis interview
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Michael Lewis
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Going Infinite
- Michael Lewis -SAID-> FTX
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Going Infinite, Michael Lewis, Alameda, Sam Bankman-Fried was a terrible boyfriend, Conspiracy to commit fraud, Risk-neutral, Gambling addict, Staff
revolt within Alameda Research over lost millions and general financial chaos, money laundering, Ellison, Sam Bankman-Fried, fraud, The Verge
</item 5>

<item 6>
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
</item 6>

<item 7>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
That was the basis of the worst of her testimony — and the spreadsheet from hell.

Ellison said she’d prepared the spreadsheet at Bankman-Fried’s request in the fall of 2021 and shared it with him. The point was risk analysis around paying
back Alameda’s loans if they were abruptly recalled by Genesis, their lender. Bankman-Fried wanted to use $3 billion for venture investments, so Ellison was
ballparking what that would do to Alameda’s risk. In the as-is scenario, if things went south, she figured there was a 30 percent chance they wouldn’t be able
to meet the loan recalls. If Bankman-Fried used $3 billion to make investments, there was a 100 percent chance they couldn’t meet the recalls, even with FTX
customer funds.

Echoing Bankman-Fried and Ellison’s romantic relationship, FTX was keeping cozy private ties with Alameda yet publicly holding it at arm’s length

The problem here wasn’t really the math, which seemed pretty arbitrary.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Sam Bankman-Fried was a terrible boyfriend, Ellison, FTX, Sam Bankman-Fried, Genesis, The Verge
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda Research
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda Research
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda Research
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda Research
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda Research

Entities: Alameda Research, Sam Bankman-Fried was a terrible boyfriend, $65 billion line of credit, Ellison, FTX, Sam Bankman-Fried, The Verge
</item 8>

<item 9>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
The testimony set up an argument for the prosecution to bring in those tapes. The judge ruled in favor of the jury hearing the recordings and we briefly
recessed.

That was when Bankman-Fried’s mother approached the defense lawyers.

The late afternoon was short and snappy. Christian Drappi, a former Alameda software engineer who looked like a handsome funeral director in a black suit and
tie, testified briefly to set up the tape. When Changpeng Zhao, the CEO of Binance, announced on Twitter that he intended to acquire FTX, Ellison confessed the
theft of customer funds to him and a few other employees, Drappi said. The all-hands took place the following day, and was secretly recorded by an employee
who’d joined Alameda three days before.

In the recording, Ellison did indeed confess to stealing customer funds with Bankman-Fried’s approval. Drappi said he resigned less than 24 hours after the
meeting. The recording was later sent to Drappi, who sent it to the government.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda

Entities: Alameda, Ellison, FTX, the government, Sam Bankman-Fried, Christian Drappi, all-hands, recording, theft of customer funds, How is it still getting
worse for Sam Bankman-Fried?, Binance, Changpeng Zhao, Twitter, The Verge
</item 9>

<item 10>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, Alameda, fiat@ftx account, testifying to Congress, Sam Bankman-Fried didn’t ask where the $8 billion went, conspiracy, customer
funds, Congress, FTX, Sam Bankman-Fried, The Verge
</item 10>
=== END ITEM id=30b86cd61a8ddaf2 ===
