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

=== ITEM id=9c95449f5a61ad7c ===
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
The end result of these tactics, the FTC argues, is elevated prices across the web. The company steadily hikes the fees it charges sellers and prevents them
from discounting on other sites, so sellers often inflate their prices off of Amazon, creating an “artificial price floor everywhere,” according to the
complaint. The FTC is seeking to hold Amazon liable for allegedly violating anti-monopoly law, though it has not yet outlined the specific remedies it believes
would best resolve its concerns. In antitrust cases, remedies are often determined only after a court finds the defendant liable. In a blog post, Amazon general
counsel David Zapolsky said third-party sellers set their own prices on the marketplace. The company also invests in tools to help sellers offer “competitive
prices,” he said. “Even with those tools, some of the businesses selling on Amazon might still choose to set prices that aren’t competitive,” Zapolsky said.

Graph facts:
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Eastvale
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Watchara Phomicinda
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Federal Trade Commission
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Fred Ruckel
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> MediaNews Group

Entities: David Zapolsky, third-party sellers, anti-monopoly law, Amazon sellers sound off on the FTC's 'long-overdue' antitrust case, FTC, CNBC, Amazon
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
Graph query row:
article: ASX set to drop as Wall Street’s September slump deepens
publisher: The Sydney Morning Herald
organization: Betashares
reported_product: None
reported_price: None
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 10>
=== END ITEM id=9c95449f5a61ad7c ===

=== ITEM id=2c1a4028749609f8 ===
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
</item 2>

<item 3>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
"We have our lists and we like to get things done."

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

"I also remember the first time we met — ee were getting ready to go out and she dug in her purse and pulled out all these tubes of lipstick. I said, ‘Oh my
God. You've surpassed me.’ We both constantly put on lipstick. It's so funny."

Bazella continued, "We both like [garden] gnomes and we’re kind of collecting them now. Our mannerisms are similar. We have the same sense of humor."

"One of the main dreams is that we stay healthy and I live long enough … so we can make up some of this time." — Sherri Geerts

Bazella said she has met all four of her siblings — and also has formed a relationship with her birth father, Victor Rebeck, 84.

Graph facts:
- Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' -PUBLISHED_BY-> FOX News - Lifestyle
- Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' -MENTIONS-> switched at birth
- Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' -MENTIONS-> Minnetonka
- Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' -MENTIONS-> Sunnyvale
- Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' -MENTIONS-> adoption

Entities: Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace', Sherri Geerts, Bazella, Victor Rebeck, Fox News
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
Graph query row:
article: Every movie and show coming to Netflix in November
publisher: Polygon
entity: Mark Zuckerberg
relationship: MENTIONS
connected_to: How ego and fear fuelled the rise of artificial intelligence
page: Every movie and show coming to Netflix in November (Polygon, 2023-11-01T12:00:00+00:00)
passage: Every movie and show coming to Netflix in November (Polygon, 2023-11-01T12:00:00+00:00)
Even with Netflix’s recommendation algorithm serving you new movies, new TV shows, and original programming tailored to your viewing habits, the streaming
service’s firehose of content makes what’s coming and going difficult to parse.

It’s a great month for adult animation, with both the Scott Pilgrim animated show and Blue Eye Samurai hitting Netflix this month. Also, there’s the Squid Game
reality show — because right, that’s a thing — and a new season of The Crown. But this month, we’re highlighting The Social Network, with another new David
Fincher movie on its way to Netflix... but not for the reasons you probably think.

Editor’s Pick: The Social Network

Hey, did you know that there’s a popular fan theory that bestselling romantic comedy Red, White, and Royal Blue started off as fanfic of The Social Network?
Well, now you do!
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
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=2c1a4028749609f8 ===

=== ITEM id=4578dde46e11741c ===
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
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
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
I’ve got some shitty ex-boyfriends, but none of them made me the CEO of their sin-eater hedge fund while refusing to give me equity and bragging about how there
was a 5 percent chance they’d become the president of the United States, you know? Absolutely counting my blessings after Caroline Ellison’s first day on the
stand. I wonder how many of the nine women on the jury are doing the same.

Ellison was the head of Alameda Research, the aforementioned hedge fund, during the implosion of it and FTX. She’s already pleaded guilty to criminal charges
stemming from one of the worst romantic relationships I’ve ever heard of, and her testimony was widely anticipated before the trial. Today, that took the form
of discussing a damning spreadsheet — one she prepared for her ex and boss Sam Bankman-Fried, now the defendant in a criminal fraud trial.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Caroline Ellison
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Caroline Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Caroline Ellison
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Caroline Ellison
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Caroline Ellison

Entities: Caroline Ellison, Alameda Research, criminal fraud trial, criminal charges, Sam Bankman-Fried was a terrible boyfriend, wire fraud, conspiracy, United
States, FTX, Sam Bankman-Fried, The Verge
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=4578dde46e11741c ===
