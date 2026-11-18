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

=== ITEM id=3c85a16d265f513b ===
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
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
Two members of the billionaire family that controls Rogers Communications Inc. are suing the company, saying they’ve been frozen out of board meetings and
blocked from getting information, reigniting a bitter power struggle within one of Canada’s wealthiest clans.

Melinda Rogers-Hixon says she’s the target of a “personal vendetta” by her older brother, Rogers Chair Edward Rogers, according to a letter contained in the
court filing. The Rogers siblings were close to an agreement that would have resolved their longstanding differences, but Edward Rogers reneged on it, according
to the Sept. 20 missive by Rogers-Hixon, which was sent to Chief Executive Officer Tony Staffieri.

The court petition was filed by Rogers-Hixon and her sister, Martha Rogers, and it argues the company has been “oppressive” to them.

The documents show that the fractures inside the prominent Toronto family have never healed.
</item 1>

<item 2>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
The end result of these tactics, the FTC argues, is elevated prices across the web. The company steadily hikes the fees it charges sellers and prevents them
from discounting on other sites, so sellers often inflate their prices off of Amazon, creating an “artificial price floor everywhere,” according to the
complaint. The FTC is seeking to hold Amazon liable for allegedly violating anti-monopoly law, though it has not yet outlined the specific remedies it believes
would best resolve its concerns. In antitrust cases, remedies are often determined only after a court finds the defendant liable. In a blog post, Amazon general
counsel David Zapolsky said third-party sellers set their own prices on the marketplace. The company also invests in tools to help sellers offer “competitive
prices,” he said. “Even with those tools, some of the businesses selling on Amazon might still choose to set prices that aren’t competitive,” Zapolsky said.
</item 2>

<item 3>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
A worker sorts out parcels in the outbound dock at the Amazon fulfillment center in Eastvale, California, on Aug. 31, 2021. Watchara Phomicinda | MediaNews
Group | The Riverside Press-Enterprise via Getty Images

It was late in the day on Oct. 27, 2021, when Fred Ruckel received the dreaded automated email from Amazon . Amazon’s software had detected that Ruckel’s
popular cat toy, called the Ripple Rug, was being sold somewhere else for a cheaper price. His product would no longer be shown in Amazon’s all-important buy
box, an area of the listing where shoppers click “Add to Cart.” Ruckel is the sole seller of the Ripple Rug on Amazon, so the move all but ensured his product
would disappear from the website, costing him thousands of dollars per day.
</item 3>

<item 4>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The Shaw takeover finally closed on April 3 of this year, and the Rogers siblings engaged in talks “intended to resolve all issues between us,” Rogers-Hixon
stated in an affidavit. “I was hopeful all issues could be resolved without negative publicity.” But those negotiations collapsed last month, she said.

It isn’t clear from the court filings whether the discussions included the possibility that any Rogers family members would sell their stake in Rogers
Communications, leave their roles at the company, or exit the family trust that controls almost all of the voting shares of the public company.

“This matter should be resolved privately,” Rogers Communications spokesperson Sarah Schmidt said by email. “We’ve demonstrated that we will not be distracted
by these actions — we have significant momentum in the market, our merger is tracking ahead of plan, and we remain squarely focused on doing what’s right for
our customers and stakeholders.”
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
</item 5>

<item 6>
Power struggle in one of Canada’s wealthiest clans has sister accusing chairman brother of ‘vendetta’ (Fortune, 2023-10-12T22:22:46+00:00)
The result was that five Rogers Communications directors were fired, replaced by five allies of Edward Rogers. Soon after, then-CEO Joe Natale was sacked and
replaced by Staffieri, the former chief financial officer. Rogers-Hixon and Martha Rogers had opposed the CEO change.

In early 2022, family members agreed to temporarily shelve their disagreements, according to this week’s court filing, to avoid negative publicity as the
company sought regulators’ approval for the takeover of Shaw Communications Inc. The C$20 billion transaction was the company’s biggest deal ever and among the
largest corporate transactions in Canadian history.

Negotiations Collapse

As part of that truce, Rogers-Hixon and Martha Rogers agreed not to participate in board discussions about the Shaw deal or receive materials on it, because of
an argument about Rogers-Hixon’s personal lawyer, whose firm also does work for Telus Corp., a competitor to Rogers and Shaw.
</item 6>

<item 7>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“Just like any store owner who wouldn’t want to promote a bad deal to their customers, we don’t highlight or promote offers that are not competitively priced.”
Zapolsky argued the FTC’s lawsuit could force it to stop highlighting low prices, “a perverse result that would be directly opposed to the goals of antitrust
law.”

“Long overdue” lawsuit

On Amazon’s own forum for merchants, called Seller Central, several users cheered on the FTC and said they hoped it would result in changes to the company’s
business practices. Amazon’s tense relationship with merchants has been well-chronicled over the years, with sellers expressing a range of grievances over
issues like rising fees, an arcane suspensions process, and heightened competition on the marketplace from all sides, including the e-commerce giant. “I think
it’s great, Amazon deserves it,” one person commented, adding, “More should be coming on the way.”
</item 7>

<item 8>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Fred Ruckel

Nearly three years later, Ruckel’s experience hits at the core of a sweeping antitrust lawsuit filed last week by the Federal Trade Commission against Amazon.
The agency accused Amazon of wielding its monopoly power to squeeze merchants and thwart rivals. For consumers, that’s led to artificially inflated prices and a
degraded shopping experience, the agency alleges. In the 172-page suit, the FTC said Amazon relies on an “anti-discounting strategy” and a “massive web-crawling
apparatus that constantly tracks online prices” to stifle competition. The agency said Amazon punishes third-party sellers who offer cheaper products elsewhere
by threatening to disqualify them from appearing in the buy box if it detects a lower price. Losing the buy box is an “existential threat” to sellers’
businesses, the complaint alleges. The end result of these tactics, the FTC argues, is elevated prices across the web.
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
</item 9>

<item 10>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Hart said Amazon sales comprise about 90% of his business, meaning any hit those sales take on Amazon has a considerable impact. He recalled “24 anxious hours”
in September 2022 when a third-party seller of his popular construction toy Brain Flakes listed the toy for a lower price on Target than it was offered on
Amazon.

Molson Hart, CEO of Viahart, an educational toy company that sells on Amazon. Courtesy: Molson Hart
</item 10>
=== END ITEM id=3c85a16d265f513b ===

=== ITEM id=8cd7989d5903d781 ===
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
“Sometimes it can be helpful for them to know about the dynamics that they’re stepping into and how they can best support you. Frankly, they should be aware if
there is tension or if there is stress within your family.”

(Getty Images)

Communication before the holidays goes both ways, and there are things that your family should know about your significant other too. “If there’s something
about your partner that could be surprising to your family, or that’s a notable difference from who someone you’ve brought home before, tell your relatives
that,” Zar said. “Let’s set your family and partner up for success.”

Before bringing your significant other home for the holidays, you may want to express to your family how you wish for them to feel included - such as
incorporating your partner’s family traditions into your own celebration. However, Zar advised against oversharing information before Christmas. You still want
your family and partner to connect in person, rather than hearing about each other through the grapevine.
</item 1>

<item 2>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
MICHIGAN SIBLINGS REUNITE WITH THEIR 3-YEAR-OLD BROTHER AFTER HIS 6-MONTH HOSPITAL STAY FOR CANCER TREATMENT

"I was shocked and just so confused," Bazella said. "But at the same time, I wasn’t."

"When I met my mom, I truly felt unconditional love." — Diane Bazella

Over the next four years, Bazella reached out to people who were listed on her genealogy report, trying to figure out how they could be related, but with no
success.

Finally, one woman she contacted, who showed up as a close DNA match, emailed her back.

"She reached out and said, ‘I want to figure this out with you,’" Bazella said.

It took the two women more than a year to unravel the story.

"We were thinking [that] maybe I had a different birth father … and that maybe it was one of her relatives," Bazella said.
</item 2>

<item 3>
The best shows to stream in December (The Sydney Morning Herald, 2023-11-30T22:00:00+00:00)
The spotted white beagle and his adolescent cohort – Charlie Brown, Marcie, Peppermint Patty, and many more – have always transferred easily from a panel on the
page to an animation frame, and this official adaptation has a trademark mix of wry humour, flights of fantasy, and heartwarming camaraderie. Happiness is
Holiday Traditions is a bonus special to close off the third season of the series, and it will stay true to the rule that helps make Peanuts endure: no adults
are seen or heard. Also on Apple TV+: The trailer for Family Plan (December 15), an action-comedy feature where Mark Wahlberg plays a former master assassin who
left the trade and secretly built a new life in the American suburbs as a cheerful husband and father, is goofy as all heck. The question is whether it’s
ludicrous fun or just sappy shenanigans?
</item 3>

<item 4>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
That isn’t necessarily a cause for concern, as long as you inform your significant other about the interworkings of your family dynamics.

“I think that we tend to be our best relational self when we’ve really done the work around understanding all of our past relationships,” Zar explained. “You
can tell your partner: ‘I don’t have a close relationship with my brother, and here’s why. Here are the intentional choices that I’ve made around it.’ Or you
can say: ‘Here’s my chosen family. These are the people who are important to me.’ Your significant other will see that you’ve been introspective and have
takeaways around how you’ve navigated your family’s origin.”

As long as you and your partner have discussed your relationship expectations - both before and during the holidays – introducing them to your family may deepen
your relationship and could be a step in the right direction.
</item 4>

<item 5>
The best shows to stream in December (The Sydney Morning Herald, 2023-11-30T22:00:00+00:00)
The question is whether it’s ludicrous fun or just sappy shenanigans? Wahlberg’s Dan Morgan, who has to go on a Las Vegas road trip with his blissfully unaware
family after his cover is blown, dispatching goons while carrying his youngest in a BabyBjorn and stunt driving in a family minivan are positive signs. Michelle
Monaghan (True Detective) plays Mrs Morgan, who quietly thinks her husband is a little vanilla, and the fine Irish actor Ciaran Hinds (There Will Be Blood) goes
full villain. November highlights: The Buccaneers placed a modern teen lens on Edith Wharton’s period marital drama, while Godzilla and pals returned for
Monarchy: Legacy of Monsters and space race alternate history For All Mankind reached the 21st century.

Advertisement
</item 5>

<item 6>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
"We have our lists and we like to get things done."

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

"I also remember the first time we met — ee were getting ready to go out and she dug in her purse and pulled out all these tubes of lipstick. I said, ‘Oh my
God. You've surpassed me.’ We both constantly put on lipstick. It's so funny."

Bazella continued, "We both like [garden] gnomes and we’re kind of collecting them now. Our mannerisms are similar. We have the same sense of humor."

"One of the main dreams is that we stay healthy and I live long enough … so we can make up some of this time." — Sherri Geerts

Bazella said she has met all four of her siblings — and also has formed a relationship with her birth father, Victor Rebeck, 84.
</item 6>

<item 7>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
While she has found a place with her birth parents, Bazella knows she may never solve the mystery of how she was switched at birth.

"I lost 39 years," Bazella said. "There’s a lot of grieving with that."

Still, Bazella said she believes she's found unconditional love with her mother.

"It's overwhelming for me coming from where I have been," she added.

"I always felt like I was kind of in survival mode and alone. I finally feel at peace with everything, which is really nice."

CLICK HERE TO GET THE FOX NEWS APP

As for Geerts, she said she hopes to make up for lost time.

"One of the main dreams is that we stay healthy and I live long enough, like a hundred years at least — so we can make up some of this time," Geerts said.
</item 7>

<item 8>
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
</item 8>

<item 9>
Every movie and show coming to Netflix in November (Polygon, 2023-11-01T12:00:00+00:00)
Over the course of two-episodes, discover your new favorite comedian.

Available Nov. 29

American Symphony (Netflix Documentary)

Bad Surgeon: Love Under the Knife (Netflix Documentary)

From Netflix: This shocking docuseries about one of the biggest frauds in modern medical history exposes a thoracic surgeon famous for groundbreaking
achievements in the field of regenerative medicine.

Available Nov. 30

The Bad Guys: A Very Bad Holiday (Netflix Family)

From Netflix: Mr. Wolf and his crew of animal outlaws realize they’ll need to restore the city’s Christmas spirit to keep their annual holiday heist afloat.

Family Switch (Netflix Film)

From Netflix: Jess and Bill Walker are doing their best to keep their family connected as their children grow older, more independent, and more distant. When a
chance encounter with an astrological reader causes the family to wake up to a full body switch, on the morning of the most important day of each of their
lives, can the Walkers unite to land a promotion, college interview, record deal and soccer tryout?
</item 9>

<item 10>
Woman switched at birth spends holidays with birth mom after decades-long search: 'Feel at peace' (FOX News - Lifestyle, 2023-12-15T09:30:54+00:00)
"I do remember the day I had to give her up," Geerts told Fox News Digital.

"I got to name her and I got to dress her. There was a lady there to take her away, and she was kind of shaky and it scared me because I didn’t want her to pick
her up."

Geerts said she thought about her daughter every single year on her birthday.

Now that the two have met, they've been getting together several times a year, either in California or Minnesota.

COUPLE OF 60 YEARS, SEPARATED FOR 215 DAYS AMID PANDEMIC, REUNITES IN TOUCHING VIDEO: 'I MISSED YOU SO MUCH'

And from the first time they met in person, they began discovering things they have in common.

"We both keep things in order," Bazella said. "We have our lists and we like to get things done."
</item 10>
=== END ITEM id=8cd7989d5903d781 ===

=== ITEM id=c17a51837f986f77 ===
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
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 1>

<item 2>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 2>

<item 3>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Ellison is expected to take the stand under a similar arrangement.

U.S. District Judge Lewis Kaplan presides as Gary Wang testifies during the fraud trial of Sam Bankman-Fried over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

Born in China, Wang moved to the U.S. at age 7, and grew up in Minnesota before going to the Massachusetts Institute of Technology to study math and computer
science. He worked at Google after college. Wang, who first met Bankman-Fried during high school at a summer camp, owned 10% of Alameda, while his boss owned
the other 90%. Wang told the court about the advantages that Alameda received by having code baked into FTX's software that allowed special access to the crypto
exchange.
</item 3>

<item 4>
Sam Bankman-Fried didn’t ask where the $8 billion went (The Verge, 2023-10-31T19:52:07+00:00)
Instead, Bankman-Fried directed alleged co-conspirator Caroline Ellison to repay third-party loans and went on making investments.

Bankman-Fried said that he found out about the fiat@ftx account — that’s the one that tracked how much Alameda owed FTX — in June 2022 while his senior staff
was focused on fixing the software bug. He did not, however, discover what the account was for until October. I know this sounds unbelievable, but this is his
actual testimony.

His employees told him “they were busy and I should stop asking questions because it was distracting.”

Simple questions nailed Bankman-Fried to the wall. Had he given his employees at Alameda the direction not to spend FTX customer deposits? Had he put in any
policies to prevent Alameda employees from spending FTX customer money? Did he put in place any measures at Alameda to protect the FTX customer money? No, no,
and no. Oh, but he was testifying to Congress about keeping customer funds safe.
</item 4>

<item 5>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
That was all done by Bankman-Fried, he testified. In early 2020, Wang said he discovered for the first time Alameda's negative balance exceeded FTX's revenue,
an indication that Alameda was taking customer funds. Wang said he brought this to Bankman-Fried's attention several times. In late 2021, Wang discovered
Alameda had withdrawn $3 billion from its $65 billion line of credit. Wang's compensation was a base salary of $200,000 per year plus stock. He owned roughly
17% of FTX. Even though they were co-founders, "ultimately it was Sam's decision to make" when there were disagreements, he said.
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
</item 6>

<item 7>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
Much of said funding came from customers, and Easton testified that he found that user deposits were used by both FTX and Alameda to purchase real estate, make
investments or direct funding toward political causes and charities.

SBF trial: What we learned in week two

The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to
repay debts to lenders, using them as a line of credit under the instruction of SBF. Ellison also went into detail about how a $2 billion venture fund, FTX
Ventures, was funded by money that had previously been allocated to Alameda from third-party lenders.

And though there are no charges of bribery in the case, Ellison also testified to paying Chinese officials $150 million to restore access to $1 billion in
frozen trading accounts.

SBF trial: What we learned in the week one

As expected, the opening statements in the SBF trial were spicy.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 8>

<item 9>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Those privileges ultimately resulted in Alameda owing FTX $8 billion worth of customer deposits. "We gave special privileges on FTX that gave unlimited
withdrawals on the platform to Alameda," Wang said. Alameda was allowed to withdraw and transfer those funds and had a $65 billion line of credit. "When
customers deposited USD, it went to Alameda," he said. "It existed in the computer code. Alameda could have negative balances and unlimited withdrawals." That
"bug" in the code was written by Nishad Singh, who was FTX's director of engineering, and reviewed by Wang. Bankman-Fried was calling the shots, Wang said. Wang
also told the court about a $1 million personal loan he received and a $200 million to $300 million loan in his name from Alameda that was never deposited into
his account, but rather was used to make investments into other companies on behalf of FTX.
</item 9>

<item 10>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.
</item 10>
=== END ITEM id=c17a51837f986f77 ===

=== ITEM id=ccadfdf14d4df980 ===
Gold sentences and facts:
Sentence 0: With this sale, they offer some of the best value in their class from one of the top names in the business.
  Fact 0.0: The seller offers some of the best value in its class.
  Fact 0.1: The seller offers this value from one of the top names in the business.
Sentence 1: “Each actor will have to be responsible to ensure that the market remains reasonable amid rates that are relatively low,” CMA CGM Chief Financial Officer Ramon Fernandez told reporters Friday.
  Fact 1.0: Ramon Fernandez is Chief Financial Officer of CMA CGM.
  Fact 1.1: Ramon Fernandez told reporters on Friday.
  Fact 1.2: Ramon Fernandez said each actor will have to be responsible to ensure that the market remains reasonable amid rates that are relatively low.

Retrieved items:
<item 1>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is often a boon for deals on headphones and earbuds, and this year is no different. The shopping holiday has brought a plethora of deals on all
kinds of headphones, include several top picks from our guides to the best wireless headphones, wireless earbuds, budget wireless earbuds and gaming headsets.
Other popular models from the likes of Sony, Beats, Bose and Apple are discounted as well. All of this means it's an ideal time to pick up a new set, whether
it's to give as a gift to someone on your holiday shopping list or just to replace your own aging cans. To help you get a good bargain, we've rounded up the
best Black Friday deals on headphones and earbuds we could find for 2023.
</item 1>

<item 2>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Engadget Editor Jon Turi found this pair to deliver more low-end depth than most of its peers, however. It can also last up to 10 hours per charge.

Your Black Friday Shopping Guide: See all of Yahoo’s Black Friday coverage, here. Follow Engadget for Black Friday tech deals. Learn about Black Friday trends
on In The Know. Hear from Autoblog’s experts on the best Black Friday deals for your car, garage, and home, and find Black Friday sales to shop on AOL,
handpicked just for you.
</item 2>

<item 3>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
Photograph: Technics

The Technics EAH-AZ80 (9/10, WIRED Recommends) are some of our favorite top-shelf earbuds thanks to their fantastic sound alongside great features and noise
canceling, and their ability to connect to up to three devices at once. This is one of the few sales we've seen all year, which makes their very high asking
price a lot more palatable.

Soundcore's Life Q30 are already impressive at full price thanks to noise canceling that punches well above its class, so this deal makes them something of a
steal. They add to the package with accessible and adjustable sound quality, good features, and great battery life.

Jump to a Topic: PC and Gaming Deals, Home and Kitchen Deals, TV Deals, Watch and Fitness Tracker Deals, Headphone and Earbud Deals, Smart Home Deals, Speaker
Deals

Smart Home Deals

Want to smarten up your home? See our guides to setting up your smart home, the best smart lighting, and the best mesh routers to get started.
</item 3>

<item 4>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
However, there’s no wireless charging, the battery life (five or so hours per charge) is mediocre and the IPX2 sweat-resistance rating is too weak for the gym.
Amazon’s companion app is rather bloated too. But those shouldn’t be deal-breakers for $35. We gave the Echo Buds a review score of 77 in July.

Anker Soundcore Space A40

The Anker Soundcore Space A40 is one of the best values in wireless earphones at its normal price of $79, but now it’s down to $54 for Black Friday. That ties
the lowest price we’ve tracked. The Space A40 tops our guide to the best budget wireless earbuds because it delivers most of the features we expect from
headphones that cost twice as much. It’s comfortable, its ANC is genuinely capable, its battery lasts up to 10 hours per charge, its tiny case supports wireless
charging and it can pair with two devices at once.
</item 4>

<item 5>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
That's another $50 discount and an all-time low. We gave these in-ears a review score of 88 in September and recently named them the “best for noise
cancellation” pick in our wireless earbuds buying guide. As that title suggests, the main reason to consider this pair over the Sony WF-1000XM5 is its ANC
performance, which is the strongest we’ve tested from a true wireless model. We also enjoy Bose’s spatial audio tech. But the earpieces are on the larger side,
the call quality and battery life (at six-ish hours) are just OK and there’s no support for multi-device pairing.

Beats Fit Pro

The Beats Fit Pro is back down to $159, which isn’t an all-time low but is still roughly $20 below its average street price. This is the “best for workouts”
pick in our wireless earbuds buying guide, as it packs many of the iOS-friendly conveniences of Apple’s AirPods in a sportier and more stable design.
</item 5>

<item 6>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
Sony's WF-1000XM5 (7/10, WIRED Review) source multiple generations of great sound, top features, and excellent noise canceling, all wrapped into their most
compact presentation yet. They're not cheap, so this discount is key if you're looking to be an earlier adopter of Sony's latest flagship buds.

The only issues we had with Sennheiser's HD 660S2 (7/10, WIRED Review) bass-enriched audiophile beauties were their high price and stiff clamping force. At this
price, we can live with a tighter fit, especially since the payoff is gloriously clear and richly detailed sound.

Sony's working-class noise cancelers offer solid performance. clear sound, and impressive battery life at a much more reasonable price than the brand's top
offerings. With this sale, they offer some of the best value in their class from one of the top names in the business.
</item 6>

<item 7>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
Anker's Liberty 4 NC earbuds (8/10, WIRED Recommends) are another feature-packed pair of buds with great noise canceling, and are on sale for $80 ($20 off).

The Beats Studio Pro (7/10, WIRED Review) were released just a couple of months ago. We wish the battery life was longer and that the controls and EQ options
were more robust. But this price makes them more worthwhile, and they do have excellent noise canceling.

Photograph: Bose

These noise-cancelling headphones are our favorite for the office. With advanced signal processing and four microphones built in, these will limit any sound
around you, making them great for focusing on work in a busy environment and for Zoom calls. These often fluctuate in price, but we think it's a great deal.

We like Sony's standard LinkBuds (8/10, WIRED Recommends) for the open-ear design that allows you to hear what's going on around you.
</item 7>

<item 8>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
As Sony’s priciest wireless headphones, the WH-1000XM5 had a lot to live up to following the remarkably popular WH-1000XM4. Luckily they deliver the goods with
upgraded noise canceling, a posh new design, and a more refined sound signature that adds up to a great package – especially on sale.

Bose’s aptly named QuietComfort Ultra (9/10, WIRED Recommends) arrived in October with only one notable problem: they’re very pricey. Luckily they’ve already
gotten a discount, which makes it easier to land their class-leading noise canceling, great sound, and luxuriously comfy design that’s loaded with modern
features.

Photograph: Sony

Sony's WH-1000XM4 headphones (9/10, WIRED Recommends) have been a stalwart pair of wireless headphones for years now, and there's no secret to their success.
Offering excellent noise canceling, warm and detailed sound, and a ton of modern features in a comfy design, they're among the best travel headphones you can
buy.
</item 8>

<item 9>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Like many Sony headphones, it also comes with a number of useful bonus features; “Speak-to-Chat,” for instance, automatically pauses your music when you start
speaking.

Sony WF-1000XM5

For an in-ear pair, the Sony WF-1000XM5 is also on sale for $248 at Amazon, B&H and others. (Yes, Sony’s headphone naming scheme is bad.) That matches the
lowest price we’ve seen and comes in $52 below Sony’s usual list price. This set tops our list of the best wireless earbuds and earned a score of 87 in our
review this past July. Like the over-ear XM5 above, it offers a host of useful features, a warm yet customizable sound, powerful ANC and a comfortably snug fit
that naturally isolates a good chunk of outside noise on its own. Battery life should sit around seven to eight hours with average use.
</item 9>

<item 10>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
If you prefer this kind of bulkier but easier-to-remove shape, this is an excellent value.

Anker Soundcore Space Q45

If you’re after a wallet-friendly pair of noise-canceling over-ear headphones, Anker’s Soundcore Space Q45 is worth a look at its Black Friday price of $100.
For that amount, this pair’s better-than-average ANC, softly padded design and multi-device pairing make for a strong value. Its V-shaped sound profile isn’t
especially detailed, but it can work well enough for pop music and it’s customizable through Anker’s app. Call quality is mediocre, however. This $50 discount
matches the lowest price we’ve tracked, and it's available at both Amazon and Anker's online store; just use the on-page coupon for the latter.

Audio-Technica ATH-M20xBT

The Audio-Technica ATH-M20xBT is a modest $10 off for Black Friday, bringing the wireless headphones down to $69.
</item 10>
=== END ITEM id=ccadfdf14d4df980 ===

=== ITEM id=35bd241455d4e59a ===
Gold sentences and facts:
Sentence 0: This not only allows the collection of data from complex equipment but also enables operators to track equipment performance and predict maintenance and downtimes.
  Fact 0.0: This allows the collection of data from complex equipment.
  Fact 0.1: This enables operators to track equipment performance.
  Fact 0.2: This enables operators to predict maintenance and downtimes.
Sentence 1: If a track isn’t used or taken care of for an extended period of time, it will start to wear out and eventually become completely unsuitable for any kind of racing unless it is taken apart and rebuilt.
  Fact 1.0: A track that is not used or taken care of for an extended period of time will start to wear out.
  Fact 1.1: A track that is not used or taken care of for an extended period of time will eventually become completely unsuitable for any kind of racing.
  Fact 1.2: A track that is not used or taken care of becomes unsuitable for racing unless it is taken apart and rebuilt.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
The companies aimed to accelerate Siemens Xcelerator adoption and democratize new digital twin solutions using AWS IoT TwinMaker, a service intended to make
creating digital twins that incorporate multiple data sources faster and easier.

Hyperscalers and simulation companies

Simulation companies bring specialized expertise in creating accurate, high-fidelity models of physical entities, be it machinery, buildings, or entire
ecosystems. By partnering with a hyperscaler, simulation companies can harness these cloud services’ data storage and computational power to create more robust,
responsive, and accurate data twins that can be scaled and integrated seamlessly into broader IT ecosystems.

Example: In 2023, Ansys and Microsoft announced a partnership to help customers envision digital twins on a large scale. In cooperation with Tata Consultancy
Services, Ansys and Microsoft integrated Ansys’s Twin Builder’s physics-based simulation capabilities with Azure Digital Twin and IoT data to allow Twin Builder
users to run scenarios and obtain predictions on how their systems will behave.
</item 2>

<item 3>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Example: In large motor applications, such as room-sized motors that pump high volumes of gas, oil, or other chemicals, operators need to monitor the motors’
temperatures—especially if the motors are high-powered and repeatedly started. If a motor is too hot during a restart, the components could cause serious
damage. Unfortunately, using a direct, physical sensor within a motor is often impractical, and operators must work from an assumptive time it takes for the
motor to cool (with an additional safety buffer time). This means motors can be down longer than necessary, impacting efficiency and revenue.

To address this issue, Siemens developed a prototype virtual sensor based on a digital twin. It offers a simulation of how a physical sensor would operate if it
were possible to install it within a motor. Using AR headsets, operators can see a simulation of the motor and its interior with a demonstrator superimposed
over it, and they can see the motor’s temperature.
</item 3>

<item 4>
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
</item 4>

<item 5>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Based on these observations and additional insights from our report, our current projection estimates that the global digital twin market could grow at a CAGR
of approximately 30% between 2023 and 2027.

Against the backdrop of this growing market, the report notes eight trends that help promote the advancement and adoption of digital twins. Here, we will focus
on four:

1. Digital twins deployed to meet sustainability goals

2. Digital twins employed as virtual sensors in complex conditions

3. Partnerships between cloud hyperscalers and OT and simulation specialists

4. Initiatives promoting interoperability of digital twins across multifaceted systems

Before we dive into these trends, and to provide context, it is helpful to understand how IoT Analytics defines digital twins and the digital twin market.

Defining digital twins and the digital twin market

In our March 2023 Decoding Digital Twins article, we define digital twins as a virtual model replicating the behavior of an existing or a potential real-world
asset, system, or multiple systems.
</item 5>

<item 6>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
If not, what investments are required?

2. Data integration: How compatible are our current data sources and systems with digital twin technologies, especially in terms of sensor data (both physical
and virtual/soft)?

3. Interoperability concerns: As we use multiple software solutions across our operations, how important is interoperability between digital twin technologies
and our existing systems?
</item 6>

<item 7>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
The partnership addressed similar challenges related to Industry 4.0 and Smart Manufacturing.

DTC digital twin interoperability framework

The Digital Twin Consortium (DTC), a collaborative partnership of industry, academia, and government experts, released the Digital Twin System Interoperability
Framework to “unify a nascent ecosystem of high-value, multi-vendor services that can seamlessly ‘plug into’ a multi-dimensional, interoperable system of
systems.”

DTC and IDTA collaboration

DTC and the Industrial Digital Twin Association (IDTA) announced a liaison agreement to collaborate on standardization requirements, enabling interoperability
through discussions, aligning work in horizontal domains, and collaborating on open-source projects, contributions, and reference implementations.

DTC and OPC Foundation collaboration

DTC and the OPC Foundation announced a liaison agreement to “accelerate the development and adoption of digital twin-enabling technologies,” promoting
interoperability standards and processes to advance the use of digital twins in manufacturing across multiple industries.
</item 7>

<item 8>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
Trend 3: Partnerships forged in the clouds: Hyperscalers team up with OT and simulation specialists

In the last few years, cloud hyperscalers like AWS and Microsoft Azure have introduced digital twin platforms, e.g., AWS IoT TwinMaker and Azure Digital Twins,
that allow for interconnecting various data sources and building digital twin topology. However, these companies realize they cannot deliver end-to-end digital
twin solutions by themselves and are thus partnering with OT and simulation companies to add capabilities to their networks.

Hyperscalers and OT companies

OT companies provide industrial data management capabilities and connections to millions of local physical assets. By partnering with hyperscalers, OT companies
can reach broader audiences looking for digital transformation solutions by offering typical cloud benefits like storage and computation power.

Example: In 2021, Siemens and AWS announced an expansion of their partnership, with digital twin technology being an area of focus.
</item 8>

<item 9>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
By the IoT Analytics team.

A new report from IoT Analytics highlights eight notable trends helping to advance and promote digital twins.

Four of these trends are discussed in detail in this article. These trends are shaping the future of the digital twin market and influencing investment
priorities for companies across various industries.

Key insights:

According to our Digital Twin Market Report 2023–2027, the digital twin market is expanding, with a projected CAGR of 30% between 2023 and 2027.

29% of global manufacturing companies have either fully or partially implemented their digital twin strategies. Further, job posts related to digital twins have
increased by 11% compared to October 2021, while openings for other tech topics have declined in the same timeframe.

The report notes eight notable trends helping to advance and promote digital twins, four of which are discussed in this article.
</item 9>

<item 10>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
A digital twin strategy as part of an overall digitalization plan can be a crucial capability for asset-intensive industries and needs to encompass the entire
asset lifecycle, process, and value chain from design and operations through maintenance and strategic business planning.”

Example: Aden, a Chinese integrated facility service provider, created a digital twin for one of its commercial centers in Chengdu, China. The digital twin was
designed to help facility managers inspect, maintain, and repair building assets. 3D simulations assist facility managers in visualizing, predicting, and
optimizing energy consumption, and the expected benefits include lower annual energy consumption, water usage, and waste.

Trend 2: Digital twins employed as virtual sensors in complex conditions

Virtual sensors approximate data that otherwise cannot be obtained via physical sensors, often due to physical sensors being impractical, costly, or hazardous
to employ. Companies are now building digital twins to model hardware and calculate data based on other conditions. This not only allows the collection of data
from complex equipment but also enables operators to track equipment performance and predict maintenance and downtimes.
</item 10>
=== END ITEM id=35bd241455d4e59a ===
