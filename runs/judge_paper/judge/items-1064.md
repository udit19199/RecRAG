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

This file has 1 items.

=== ITEM id=b27d40e3c2fbd070 ===
Gold sentences and facts:
Sentence 0: Both PMS and mutual funds provide expert guidance in navigating volatile markets and seizing investment opportunities that help create wealth.
  Fact 0.0: PMS provides expert guidance in navigating volatile markets.
  Fact 0.1: PMS provides expert guidance in seizing investment opportunities.
  Fact 0.2: PMS helps create wealth.
  Fact 0.3: Mutual funds provide expert guidance in navigating volatile markets.
  Fact 0.4: Mutual funds provide expert guidance in seizing investment opportunities.
  Fact 0.5: Mutual funds help create wealth.
Sentence 1: Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results, especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).
  Fact 1.0: Brown said he was surprised.
  Fact 1.1: Brown was surprised that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results.
  Fact 1.2: Kroll was appointed to consider the reasonableness of Citrin Cooperman's assumptions for HSF's previous set of results.
  Fact 1.3: HSF's previous set of results runs to end of March 2023.

Retrieved items:
<item 1>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
“[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try and pin the
blame on [Hipgnosis Songs Management].” Christopher Brown, JP Morgan

Concluded Brown: “[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try
and pin the blame on [Hipgnosis Songs Management].

“Our view is that a breakdown in the relationship between Board and manager would not be optimal for maximising shareholder value.”

Brown further noted that JP Morgan would “not be surprised to see Citrin Cooperman resign, having been very publicly undermined by the [HSF] Board today”.

JP Morgan retained its ‘overweight’ rating against HSF’s share price “in spite of yet another [HSF] own goal”.
</item 1>

<item 2>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
When the risks are higher, you must apply a higher margin of safety before choosing the stock.

Another thing to bear in mind when it comes to EV-based valuation — any change in EV will impact only the equity value. For example, if EV of a company is ₹100
and it has ₹50 in debt, then the value of its equity is ₹50. In this case, if you think EV can increase by 10 per cent, the value of equity will increase by 20
per cent, and the reverse is also true. The higher the leverage, more the impact of change in EV on the value of the stock (both ways).

Keep a weather eye on Ketchup Economics

Given the fact that relative valuation is based on comparison with peer multiples, investors must take measures to check whether the peer multiples are
reasonable as well. An overvalued peer cannot be used to justify expensive valuation in one stock. This is what economist Larry Summers termed as Ketchup
economists/economics.
</item 2>

<item 3>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
HSF’s investments in seasonal Christmas ‘evergreens’ continue to pay off too, with cuts on Michael Buble’s Christmas album and – most famously – a share of
Mariah Carey’s perennial megahit, All I Want For Christmas Is You.

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 3>

<item 4>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
We worked with them on a bold and innovative Lil Yachty alternative album, and we’re all looking forward to new music from Lil Baby next year. Offset not only
made a great album – I expect Grammy nominations for him next year – he really puts the work in, day-in, day-out.

“Music is music, and genres shouldn’t be defined by color. In fact, this business really needs to change its definition of “pop” music. There seems to be real
confusion there; we should be calling any music that streams in volume, “pop” music.”

I knew that if we were going to make significant inroads with hip-hop and R&B company-wide, we needed experts in those areas and fresh perspective on our
executive team.
</item 4>

<item 5>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
[Motown was spun out in early 2021, then returned to CMG in early 2023].

Second, we took a hard look at every aspect of our business. We revamped our entire organizational structure, reset our value system, and even replaced certain
terminology that was outdated. All to meet the needs of the modern music business. Arjun was key to this process; his expertise really drove a new dynamic in
how marketing, creative, international, digital and commerce now interact.

“Today, literally every plan is tailored for a specific artist or project — not pulled from a well-worn playbook — and our focus is on finding and creating
opportunities that will have an impact, no matter where an artist is in their development.”
</item 5>

<item 6>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Being involved in this has been a real career highlight for me, without a doubt.

If I could give you our patented MBW magic wand to change something about today’s business right here and now, what would it be and why?

We could be talking for hours about this, Tim, and there are a few things out there I’d like to change with that wand! But I’ll spare you an exhaustive rant and
focus on one issue.

We need more diversity at all levels and within all facets of the music industry, particularly on the label side.

“along with doing what I can to increase the number of women at our company. There just aren’t enough of us, particularly within the most senior ranks at the
labels.”

Diversity is a superpower.
</item 6>

<item 7>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Their business valuations grow exponentially as they mature and enter the late-stage or pre-IPO stage. The mid-stage PE fund allows investors to participate in
the growth of such companies.

c) Late-stage or pre-IPO funds: Pre-IPO funds allow investors to invest in mature companies preparing to list on the stock exchange. Such funds build a
portfolio of companies about to go public and plan to exit them after their anticipated blockbuster listing, which can magnify returns in a relatively short
period. Such funds gained immense popularity after the 2020 stock market rally that witnessed the blockbuster listing of some of the popular IPOs of B2C
companies.

d) Hedge funds: These funds invest in both publicly traded and unlisted stocks. They manage their funds using long-only and long-short strategies, with the
latter allowed to use complex derivative techniques to hedge the portfolio. The specialty of long-short funds is that they try to optimise returns in every
market situation. Hence, they tend to rise and fall lower than their peers.
</item 7>

<item 8>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Today (September 28), HSF informed its investors that “credible third parties are already engaged in this go-shop process” following its announcement on
September 14.

As we’ve previously reported, there are potential obstacles in the way of these third parties’ bids. These include the fact that HSM (on behalf of HSC) has a
‘matching right’, meaning that if HSC decides to match the higher bid of a rival, HSF has to sell the catalogs to it – rather than the rival.

In addition, any ‘superior’ offer would have to exceed the aggregate cash net proceeds that would ultimately reach HSF’s coffers should HSC’s $440 million offer
be executed. (Read more about this condition through here.)

Still, it’s a theoretical possibility that a non-Hipgnosis company swoops for the 29 catalogs from HSF, and gazumps HSC in the bidding process.
</item 8>

<item 9>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Not to mention this line in various HSF company reports to date: “The [HSF] Board is ultimately and solely responsible for overseeing the valuation of the
Company’s investments in music catalogues and has appointed [Citrin Cooperman] to perform this specialist work.”

WHAT’S really going on?

It’s a fair bet that Hipgnosis Songs Fund’s board – which last week appointed activist investor Christopher Mills as a non-exec Director – is making something
of a ‘hidden’ statement to shareholders with today’s announcement.

It may be a coded attempt by HSF’s board to gnash teeth towards HSM/Mercuriadis’ ‘call option’ – which, as MBW has pointed out before, leaves the exec with the
exclusive power to sweep in and acquire HSF’s portfolio under several different circumstances.
</item 9>

<item 10>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Partners of CMG have made their own headlines in the past 12 months, too: Quality Control – which inked that JV with Motown back in 2015 – was sold to the
Scooter Braun-led HYBE America for $300 million in February; Elliot Grainge’s 10K Projects – with whom CMG partners for Ice Spice – was 51%-acquired by Warner
Music Group in September.

(CMG continues to work exclusively with Quality Control on acts such as Quavo, Lil Yachty and Lil Baby, while Ice Spice was ‘carved out’ of 10K’s Warner deal
and remains signed in partnership with CMG on an ongoing basis.)
</item 10>

<item 11>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
WHAT’S HAPPENED?

MBW Explains is a series of analytical features in which we explore the context behind major music industry talking points – and suggest what might happen next.
MBW Explains is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

The drama at Hipgnosis Songs Fund isn’t letting up for the Holidays, it seems.

This morning (December 19), the company’s board – now led by Chairman, Rob Naylor – issued a statement announcing that HSF would no longer, as previously
expected, be issuing its latest financial results today.

Instead, the results (covering the six months to end of September 2023) are now expected to be published sometime before the close of December 31.

The reason for the delay?
</item 11>

<item 12>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
These products range from tools for “creative or marketing assistance”, to tools that “potentially give us the ability to better protect artist content or find
it when used in an unauthorized fashion” as well as “brand new products that have never been launched before”.

Kooker said that some of these conversations “also include potential equity investments which would accelerate the development of these companies”.

The Sony Music exec cited, as an example, a recent generative AI-powered project around a reissue and remix of an album. Although Kooker didn’t name it
specifically, SME did recently announce a generative AI project with The Orb and David Gilmour, which was a partnership between Sony Music Entertainment, Legacy
Recordings AI company Vermillio

“These artists are known for their cutting-edge experimentation in music,” said Kooker.
</item 12>

<item 13>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
There are a few clues in HSF’s latest public statement to this end, perhaps including its rare – not to mention conspicuous – mention of HSM being
“majority-owned by funds managed and/or advised by Blackstone“.

In addition, today’s statement is also a clear signal from the HSF board that it has understandable concerns over the now-infamous ‘discount rate’ used by
Citrin Cooperman to value HSF’s portfolio (as MBW covered in depth through here).

Citrin Cooperman continues to stick to an 8.5% discount rate in its valuations, despite interest rate rises over the past 12-18 months – and has long refused to
move this 8.5% figure upwards to, say, 9% or 9.5%.

If Citrin Cooperman ever did raise this discount rate, the independent valuation of Hipgnosis Songs Fund’s portfolio would, in turn, instantly fall.
</item 13>

<item 14>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
This has been evidenced by the 44% total return, including right to income, transaction fees and expected taxes, realised since acquisition on the [$440
million] Transaction.

“Furthermore, the Board and the Investment Adviser are committed to ensuring that this value is achieved for shareholders by a re-rating of the share price and
are determined to deliver on the ongoing opportunity of the Company.”Music Business Worldwide
</item 14>

<item 15>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
The PE (price to earnings ratio) of a stock is 18. Is that cheap or expensive? Well, it depends. On what? Many things! Welcome to the world of relative
valuation.

The findings from research done a decade earlier by Aswath Damodaran indicated that around 85 per cent of valuation done by equity research analysts and 50 per
cent of corporate M&A transactions were based on relative valuation. So, what exactly is this relative valuation?

It is the art of attempting to determine the fair value of a stock based on how it is valued relative to many factors. So, whether the PE of a stock is 15 or 18
or 30, the only way you can make an assessment of whether it is cheap or expensive is by analysing this one number — against its growth prospects, peer company
multiples, industry trading range, historical average PE, the prevailing interest rates, etc.
</item 15>

<item 16>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
However, to get the clearest view of the direction the music industry would like to see things head, it might help to read the submission from the National
Music Publishers Association (NMPA).

The NMPA’s submission, dated October 30, 2023, pulls no punches.

It starts off by stressing that its membership – US music publishers major and independent – are “not opposed” to AI.

“There is widespread belief in the music industry that great benefits could come from generative AI systems that can assist human creators,” the NMPA states.

But then it goes for the jugular: “However, the development of the generative AI marketplace is marked by breathtaking speed, size and complexity. Hindsight may
well prove that there is no hyperbole in saying that generative AI is the greatest risk to the human creative class that has ever existed.”

“Hindsight may well prove that there is no hyperbole in saying that generative AI is the greatest risk to the human creative class that has ever existed.”
</item 16>

<item 17>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
NMPA

Furthermore, it contends that any argument that an AI algorithm was trained on copyrighted material for “noncommercial” or “research” purposes should be “viewed
skeptically.”

Any business could claim it’s training AI for research or non-commercial purposes, “then shift entirely to commercial exploitation, leaving the creators of the
copied works with no compensation,” the submission argued.

AI training also fails the fair use test on the second factor, as generative AI can be used to create “expressive” works of music, and on the third, given that
AI training means the entirety of a copyrighted work is ingested, “although there is no inherent need for that,” the NMPA states.

AI training also fails the final test because “AI can be used to generate works that compete in the marketplace with the copied works, thereby reducing revenue
from existing licensing markets as well,” the NMPA stated.

The issue of fair use could prove to be one of the key battlegrounds between AI developers and copyright holders.
</item 17>

<item 18>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Equity multiples

If you check the balance sheet of any company, the liabilities side is primarily made up of two line items — shareholder funds and borrowings. To the contrary,
a Profit and Loss statement has many line items — starting with revenue, followed by variable costs and then fixed costs and taxes. The only claim that equity
shareholders have for distribution or for appropriation amongst themselves is what is left after paying the taxes — net profit.

So, if you are doing relative valuation based on equity multiples, it is important to ensure you divide the equity – ie price per share or market cap of the
company by line items that apply to equity shareholders.

Based on this, there are only 3 items that can be attributed to equity shareholders — net profit from P&L statement, book value or shareholder funds from the
Balance Sheet, and the free cash flows to equity (FCFE) from the cash flow statement. Any relative valuation multiples will be linked to these three items one
way or the other. Any other metric could be wrong and misleading.
</item 18>

<item 19>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
So yeah, we are in that business.

“We are in the business of administering [rights], which means collecting revenue from thousands of platforms around the world; very complicated matters. But
we’re also in the business of ownership.”Music Business Worldwide
</item 19>

<item 20>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
One factor potentially making the future termination of HSM as HSF’s investment advisor less likely? MBW understands that, in such a scenario, HSM would have a
first-refusal option to acquire HSF’s copyright portfolio – and could potentially turn to Blackstone (i.e. the backers of Hipgnosis Songs Capital) to find the
money to pull off such an acquisition.

Why the $440 million offer is connected to the date of HSF’s ‘continuation vote’

While HSF shareholders are considering the $440 million offer from HSC to buy the 29 catalogs, other non-Hipgnosis companies are also being invited to make bids
for the same portfolio.

Those bids are enabled by a ‘go-shop’ clause in the HSF/HSC proposal that allows the HSF board to consider these non-Hipgnosis approaches for a limited period
(until 11.59pm on October 23, to be precise).
</item 20>

<item 21>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
These amendments would give HSM a 12-month notice period as HSF’s investment advisor should the IAA be terminated at any point following the upcoming 2023
‘continuation vote’.

Another clause in the new IAA: If HSF’s share price stands at an average discount to ‘operative NAV’ of 10% or more across the month of January 2025 (with that
NAV determined at the time of publication of HSF’s interim report for the period to September 30, 2024), the HSF board says it intends to serve notice to
terminate the IAA with HSM.

However, HSF’s board has also given itself the freedom to un-terminate the IAA during the entire notice period that would follow (e.g. if HSF’s board serves
12-month notice on HSM at the end of January 2025, it would have until the end of January 2026 to withdraw its termination decision).
</item 21>

<item 22>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
But there are many times when company profits are low, but its operational performance ie EBITDA is not bad. In such cases, as revenue and EBITDA continue to
grow, operating leverage will result in PE growing much faster than the above two line items.

For example, take the case of Bharti Airtel: at the start of FY22 (April ‘21) its trailing PE was negative/invalid due to losses in FY21. The stock would have
been avoided if one went by PE alone. However, on trailing EV/EBITDA basis, it was trading at 10 times EV/EBITDA. When this was considered against its estimated
next 2 years EBITDA CAGR of over 20 per cent, the valuation, combined with qualitative factors, made it an interesting buy. Since then, net profits have
improved substantially as operating leverage played out.

At bl.portfolio we had recommended a buy on the stock based on this in April 2021 and the stock has returned 80 per cent since.
</item 22>

<item 23>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
The $440 million price also reflects a +51% premium on the value of the 29 catalogs implied by Hipgnosis Songs Fund’s public market cap for the 30 days ending
September 13.

What happened today?

Amongst the news confirmed by HSF today (September 28) is that the firm’s Chair of its board since 2018, Andrew Sutch, 73, is stepping down, both as HSF board
Chair and as a director of the company.

HSF says it’s begun a process to recruit a new Chair; Sutch (pictured inset) will leave his post as and when this person is hired, or in any case before HSF’s
AGM next year (in Q4 2024).

In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five
directors.
</item 23>

<item 24>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Worth remembering: The HSF board has known about Citrin Cooperman’s latest valuation (as of September 30) for at least the past eight days.

We know this because, as mentioned, on December 11, HSF announced that the $23.1 million price of an agreed HSF ‘non-core’ asset sale represented a 14.2%
discount vs. Citrin Cooperman’s latest valuation of the portfolio (i.e. the HSF board had seen this latest valuation).

Yet it took until today (December 19) for HSF to alert its investors to its dissatisfaction with this valuation.

WHAT does JP Morgan make of it all?

One of the closest watchers of Hipgnosis Songs Fund in recent years has been JP Morgan and its analyst Christopher Brown.
</item 24>

<item 25>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“Licensing musical works before training use is both required and practicable,” the NMPA’s submission states, adding that “there are well developed processes in
place for technology ventures to obtain free market licenses on a large scale.”

And indeed it would have to be on a “large scale,” given that AI algorithms such as large language models are trained on countless millions or even billions of
pieces of digital data.

“Transparency and recordkeeping requirements are needed to disincentivize infringing activity and to support enforcement activities by copyright owners.” NMPA

The NMPA notes that work on licensing “is already underway,” pointing to news reports earlier this year that UMG, along with Warner Music Group, are in talks
with Google to secure licenses for artists’ vocals and musical melodies to create AI-generated songs.
</item 25>

<item 26>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Music is music, and genres shouldn’t be defined by color. In fact, this business really needs to change its definition of “pop” music. There seems to be real
confusion there; we should be calling any music that streams in volume, “pop” music.

Our philosophy is that great ideas transcend genre. We have experts in every “genre” but we all share, collaborate and respectfully challenge each other. In
fact, I don’t even like the word, “genre.” Let’s change it!

Be honest: When you first heard “Unholy” – from that well-known smooth soul singer Sam Smith – what was your reaction? Did you know it was a smash hit?

Sam drove this record, from its creation through the decision for it to be the first single.
</item 26>

<item 27>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Brown noted that a 0.5% rise in Citrin Cooperman’s discount rate (i.e. raising it from 8.5% to 9.0%) would “reduce the fair value of the [HSF] portfolio by
7.9%, which on a leveraged basis is [approximately a] 10% reduction in NAV [Net Asset Value]”.

Should HSF’s board go down this route, he noted, he expected “the [HSF] shares to weaken… given that this represents another ‘unforced error’”.

“This is an early blow to the credibility of the new [Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent valuer,
Citrin Cooperman.”
</item 27>

<item 28>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
If a ‘superior proposal’ (versus the $440 million offer from HSC) is received during the ‘go-shop’ period, said HSF today – whether or not HSC uses its
‘matching right’ to beat it – it may be necessary to adjourn the HSF AGM (and therefore the 2023 ‘continuation vote’) into November.

In its note to shareholders today, the HSF board reiterated its belief – and Hipgnosis Song Management’s belief – that HSF owns “a unique portfolio of iconic,
culturally significant songs that will deliver strong long-term value as they benefit from the structural tailwinds in the music industry”.

The note added: “Furthermore, the [HSF] Board believes that the Investment Adviser’s approach to Song Management should enable the Company to outperform the
wider music market.
</item 28>

<item 29>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
Despite both stocks lagging behind the broader market index, Seeking Alpha's Quant Rating system has approached them cautiously, assigning a Hold rating. In
contrast, Wall Street analysts express optimism, giving both stocks a Buy rating.

Given the prevailing market conditions and the upward trend in crude oil prices, analysts anticipate Y/Y growth in earnings per share and revenue for CVX, while
expecting a decline for XOM.

Also reporting: WisdomTree (WT), AbbVie (NYSE:ABBV), Phillips 66 (PSX), Colgate-Palmolive Company (CL), Sanofi (SNY), T. Rowe Price Group (TROW), Stanley Black
& Decker (SWK), Charter Communications (CHTR), Newell Brands (NWL), Xcel Energy (XEL), Aon (AON), AutoNation (AN), CBRE Group (CBRE), and more.

Related stock stories:
</item 29>

<item 30>
Japan’s prime minister Kishida says: ‘reassess our economy’ (Financial Times, 2023-10-01T23:01:09+00:00)
Japanese prime minister Fumio Kishida will appear this week at a series of events aimed at attracting foreign investment © AFP via Getty Images

There are distinctive catalysts for that, he adds — noting a recent push by the Tokyo Stock Exchange for companies to improve their capital efficiency, and the
high profile decision of Warren Buffett to invest heavily in five of Japan’s largest trading houses.

Hitchens says: “Is there some healthy scepticism? Yes. However, global investment firms know that they need to invest and build their offerings in Japan to meet
the growing appetite, and that they need to pay attention to an equity market at a 33-year high.”

Topix, Japan’s main stock index, has risen more than 25 per cent this year, outperforming almost all the world’s developed market indices and, as some
conference attendees admit, creating a “fear of missing out” on a rally that may have significantly further to go.
</item 30>

<item 31>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
More broadly, you might ponder a bigger question.

When the day comes that YouTube asks the world’s biggest superstars to wholeheartedly embrace ‘Dream Track’, will Google’s recent US Copyright Office filing
give said superstars – and their record companies – pause for thought?

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 31>

<item 32>
Psagot worth 80% less since Rani Zim bought it (Globes English | Israel Business Arena, 2023-11-05T13:10:32+00:00)
Immediately after acquiring Psagot’s business (which mainly consisted of management of mutual funds and customer portfolios) Zim told "Globes": "Our headline at
Psagot is to restore its former glory. We haven’t come to make a killing in one or two years, and I don’t want to start selling the business on to other
entities… we’ll invest in the people, in the activity, in the organization, in the market, and we’ll form collaborations with many entities in the economy. We
want to bring new activities and a different aroma to the market."

But, as reported six months ago, Zim has started to look for someone to buy his entire stake in the investment house, or at least part of it. One of the names
mentioned in the press is that of Group Psagot CEO Yaniv Bender, who currently holds 7.4% of the company, as someone who could participate in such an
acquisition.
</item 32>

<item 33>
Check Point sees 'fantastic' year ahead even as rival report disappointing billings (Seeking Alpha, 2023-11-18T13:56:00+00:00)
"We have the right level of redundancies and business continuity plans in place around the world," Hollenbeck said. "So we're okay."
</item 33>

<item 34>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
— Ryan Selkis 🪳 (@twobitidiot) October 2, 2023

Shameful that Michael Lewis is defending SBF. “They had a great real business. If no one had cast aspersions on the business, if there hadn’t been a run on
customers deposits, they’d still be making tons of money” – Michael Lewis That's not true. They are not a bank, they're… pic.twitter.com/vRKBbgzITe — Dan Held
(@danheld) October 2, 2023

But even so, Noff is defensive: She also believes crypto is no worse than the stock market, which can also be manipulated and mishandled.

“Even when you identify corruption, I don’t think that means we don’t believe in the system in question,” she added.

Yet Lewis’s picture is a little less stark from the looks of the 60 Minutes episode.
</item 34>

<item 35>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
its share price).

For example: On September 14, HSF’s market cap stood at approximately GBP £1.125 billion, which represented a 40% discount on the firm’s operative NAV of USD
$2.316 billion as of the end of March this year.

(Said ‘operative NAV’ was calculated by an independent valuer – Citrin Cooperman – using a discount rate of 8.5%.)

This is where the importance of that potential $440 million transaction comes in.

As MBW reported earlier this month, Hipgnosis Songs Fund’s shareholders are currently considering a proposal to sell 29 of the company’s catalogs to another
Hipgnosis company – the Blackstone-backed Hipgnosis Songs Capital (HSC) – for USD $440 million.
</item 35>

<item 36>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
These annuities are regularly distributed by funds to the investors, giving them the characteristic of yield. When the fund reaches maturity, the assets may be
sold to another fund or infrastructure trust, resulting in capital gains in overall fund returns.

g) Venture debt funds: Start-up promoters frequently want to raise money without selling equity. As start-ups, they are unable to approach banks and mutual
funds. This is where venture debt funds, which specialise in providing credit to start-ups, come in handy. It goes without saying that such AIFs help these
start-ups raise funds at high rates. The higher returns of such funds and risk management appeal to risk-taking, yield-hungry investors.

The above fund list is not exhaustive but highlights some of the unique investment opportunities available to HNIs through AIFs. Many more such options are
available to investors, such as special situation funds, real estate yield funds, and so on.

Along with the investment opportunities presented by AIFs, some other alternatives available to HNIs are:

2.
</item 36>

<item 37>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
When other assets get more attractive one must assess scope for valuation change in equities as well. For example, interest rates and bond yields in many
developed markets, including the US, are at their highest levels since 2007. This means investors may have to assess current valuations not just against last 10
years’ valuations, but of prior periods as well.

Why? Near zero interest rates and low bond yields made stocks with earnings yield of mere 1-2 per cent attractive for an FPI — for example stocks such as Asian
Paints, Nestle India and ABB India which today trade at PE of 65, 83 and 85 times respectively. This implies earnings yield of 1.5 per cent for Asian Paints and
1.2 per cent for Nestle and ABB. Such earnings yields of these stocks were fine for FPIs to hold when compared to zero interest income in their domestic market.
But now, with US bond yields offering 5 per cent, these may not be so attractive even when considering the growth prospects.
</item 37>

<item 38>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
However, not all funds are required to collect ₹1 crore in a single transaction. They can collect the money in tranches (called drawdowns) over 2-3 years. The
drawdown mechanism makes these funds easy on the wallet. AIFs have amassed an AUM of ₹8.30-lakh crore in the 11 years since their inception, demonstrating the
shift of HNI money to this asset.

The following are some of the investment opportunities available only to HNIs and presented by AIFs:

a) Start-up funds: Angel investors looking to invest in ideas that have the potential to grow into full-scale businesses can do so through start-up funds, which
provide a unique platform for bringing promoters and investors together. These funds handpick start-ups after extensive due diligence and build a portfolio with
the potential to multiply capital several times.

b) Mid-stage private equity (PE) funds: Investors who are hesitant to take the risk of investing in start-ups can invest in companies in the middle of their
evolution cycle.
</item 38>

<item 39>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Let us look at a few examples of such investment opportunities:

1. Alternative Investment Funds (AIFs)

These funds were launched in 2012 after the market regulator SEBI issued guidelines for their management. The guidelines enable AIFs to identify and make
investment opportunities available to investors in the public and private markets. AIFs, for example, can invest in listed and unlisted equities via private
equity, infrastructure assets, venture debt, stressed assets, start-ups, and so on.

As a result, the term “alternatives” came about. The guidelines have divided these funds into three categories to make it easier for investors to understand the
fund’s investment objective. These categories are I, II, and III.

A noteworthy feature of AIFs is that their returns are generally uncorrelated with publicly traded equity or debt markets, providing a natural hedge against
volatility and genuine portfolio diversification.

Because of the minimum investment threshold of ₹1 crore, AIFs are only available to HNIs.
</item 39>

<item 40>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Income multipliers and principal protection are common features of equity-linked MLDs. In the former case, the MLDs can generate a return that is a multiple of
the underlying index’s return. This increases the likelihood of a higher overall return for a more minor market movement. The latter option safeguards the
principal (up to the face value of the debenture) in the event of an adverse market movement. This feature allows even investors with a moderate risk tolerance
to easily participate in the market.

From the discussion above, it is amply evident that HNIs have various alternative avenues to grow their wealth. The distinct value proposition of AIF, PMS, and
MLDs, with unique features, distinctive risk-return profiles, and low correlation with traditional asset classes, makes them genuinely deserving of a place in
the portfolio of every HNI. Before investing, investors should conduct thorough research to understand how these investment options work. It’s important to
consider the past track record and management team before making investment decisions.
</item 40>

<item 41>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
NMPA

The submission adds: “Even more alarming is that we do not know how long the window is to act before it is too late. We therefore implore the [US Copyright]
Office to support proactive protections for human creators and, where there is uncertainty, to err on the side of protecting human creators.”

That position might not come as a complete surprise from the NMPA, which earlier this year joined the fledgling Human Artistry Campaign, whose goal – in
broadest terms – is to ensure that AI doesn’t replace or “erode” human culture.

Yet the NMPA’s submission is hardly a Luddite diatribe against high tech; rather, it advocates for regulations and principles that it sees as the right approach
to ensuring that the interests of musical artists – and music rights holders – don’t end up subjugated in the frenzy to build our brave new AI-powered world.

Here are some of the key arguments the MNPA made in its submission…

1.
</item 41>

<item 42>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
It’s important to consider the past track record and management team before making investment decisions. Consultation with investment advisors who are
knowledgeable in this area can undoubtedly be beneficial.

The author is President & Head, Nuvama Wealth
</item 42>

<item 43>
‘Is this worth it?’: The difficult questions the Shameless founders have had to ask each other (The Age, 2023-10-07T18:00:00+00:00)
How worried are they about getting it wrong? Turns out, not very. Andrews and McDonald know what good advice looks like, and how to give it. “I think the best
advice is often not a statement – it’s a question,” says Andrews. “It’s listening to what someone’s dilemma is, then posing a question that helps them
re-frame.” As two women who are constantly being asked for advice, what are the questions they ask themselves? “What’s the next thing?” says Andrews, without
hesitation. “I’m constantly wondering, what’s the next show that isn’t out there.”

Advertisement
</item 43>

<item 44>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
which they were not open to before. And they just created this incredible tidal wave. And these things happen in waves, ebb and flow.

“So I think it likely is slowing down a little bit, but that’s okay. Because it makes the multiples come down a little bit more.”

“I think what happened in the last 15 years is incredible. They had literally zero people in the subscription model and now we have 700 million people in the
world in the premium experience.” Robert Kyncl, Warner Music Group

In other words, Warner Music can go out and buy more if the multiples are low enough?

“Of course. We are in the business of buying catalogs, obviously. We have the publishing catalog of David Bowie, for instance, which we bought a few years ago.
So yeah, we are in that business.
</item 44>

<item 45>
Psagot worth 80% less since Rani Zim bought it (Globes English | Israel Business Arena, 2023-11-05T13:10:32+00:00)
"Rani and Psagot suffered from a lot of bad luck, because sometimes it’s all a matter of timing," a senior capital markets source told "Globes". "Since he
bought Psagot, three significant things happened that affected the company. First of all, interest rates rose, and if there’s one thing that a financial marker
doesn’t like, it’s interest rate hikes, particularly in the case of a leveraged deal."

To finance the acquisition, Zim took a loan of NIS 255 million from Mizrahi Tefahot Bank, which he offloaded onto the acquired investment house. NIS 120 million
is still outstanding.

"A second factor," the source continued, "is the judicial overhaul promoted by the government in the first nine months of the year, which didn’t exactly
contribute to the financial markets, with a great deal of money leaving Israel.
</item 45>

<item 46>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
And it’s totally fine to be an investor relying on relative valuation to make your savings work for you, provided you approach it the right way and ensure
adequate risk management. In relative valuation, the broader you assess the valuation, ie assess valuation multiple against more parameters, the better the
perspective you get.

In simple words, one way to understand the difference between absolute valuation (explained in last week’s edition) and relative valuation is this – absolute
valuation is made up of an interesting story that is backed by credible and validated numbers, while relative valuation is made up of an interesting number
backed by a credible and validated story.

What are those numbers in relative valuation? Every number in relative valuation is the result of a number A divided by a number B. For example, PE is price of
the stock divided by its earnings. These are all numbers that are used to compare against similar metrics of other stocks and assess the merits of the stock.

Before we proceed on using relative valuation to pick stocks, it is important to understand that you can do relative valuation using equity multiples or
firm/enterprise multiples.
</item 46>

<item 47>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
the multiples that Merck Mercuriadis paid to acquire it?

Such questions will only be answered when Hipgnosis Songs Fund’s board elects to release the company’s six-monthly financial results to end of September 30.

The most important KPI within those results, in MBW’s view: the company’s Pro-Forma Revenue, and how it compares, like-for-like, vs. the previous year’s
equivalent period.

In the meantime, back in the ‘real world’ of the music biz, HSF is actually enjoying quite a fanciable Q4.

The company is benefitting from its cuts on hit albums including Taylor Swift’s 1989 (Taylor’s Version), via its acquired Jack Antonoff catalog, as well as
Swift’s Lover (via its acquired Joel Little catalog).
</item 47>

<item 48>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Call it our biggest, our best, our strongest—they’re all accurate.”

You referred to CMG not being in strong shape and going through turbulence when I took over, and that’s all true.

I don’t think any label group has been able to turn things around as quickly as my team and I have done in less than two years.

One decision that I’m guessing wasn’t in your control in the past few years was the ‘spinning out’ of Caroline/Virgin Music into what is now Virgin Music Group.
What have been the pros and cons of that move for you?

Virgin [first, Caroline] was a major priority for us since the launch of Capitol Music Group in 2012. We really prioritized this area, and oversaw the
transition from Caroline into Virgin directly within our company.
</item 48>

<item 49>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
This is what economist Larry Summers termed as Ketchup economists/economics. It was his sarcastic take on some of the highly paid economists and finance
professionals who are concerned only with inter-relationships between prices of different financial assets. Based on this approach, they conclude whether the
price of an asset independently is efficient or not. If two bottles of ketchup sell for twice as much as one bottle (except for minor differences traceable to
transaction costs), the market is efficient.

The risk in this – it ignores the aspect of whether one bottle of ketchup is priced rationally. Nobel Laureate Paul Krugman once explained how the US housing
bubble that precipitated the global financial crisis reminded him of ketchup economists. Before purchasing a house during the bubble, buyers carefully compared
prices with ‘prices of other houses’, but ignored whether the overall level of a home price made sense.

How to avoid pitfalls

So, what can you do to avoid pitfalls while using relative valuation? Here are a few pointers.
</item 49>

<item 50>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
She had been by Barnett’s side throughout a run of success at CMG over multiple years – via artists like Sam Smith, Halsey, Paul McCartney, and Lewis Capaldi,
as well as transformative deals like a JV between Motown (then part of CMG) and Quality Control.

In actuality, Jubelirer’s immediate ascension to running CMG was not forthcoming. Instead, Barnett was succeeded in December 2020 as Chairman/CEO of the label
group by Jeff Vaughn, a fast-rising exec who’d previously impressed as a data-and-socials-savvy A&R at Mike Caren’s Artist Partner Group (APG).
</item 50>

<item 51>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In a rather damning note issued today covering the delay in the publication of HSF’s results, Brown wrote: “This is an early blow to the credibility of the new
[Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent valuer, Citrin Cooperman.”

Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results,
especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).

Added Brown: “We think the easiest solution now for the [HSF] Board is to apply an additional discount rate to the [Citrin Cooperman] valuation to reflect
current market uncertainty, on the basis that it is probably too late to employ another valuer.”
</item 51>

<item 52>
Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office. (Music
Business Worldwide, 2023-11-23T18:48:48+00:00)
They’re stars, for sure. But in both the case of Universal Music Group and Warner Music Group, they’re – currently anyway – not the biggest megastars on either
company’s books.

You might not expect, at this early stage, for the likes of UMG to put forward Drake, Taylor Swift, or The Beatles – or for WMG to put forward Ed Sheeran or Dua
Lipa – to be guinea pigs in any early AI music experiments, even if they’re run by ‘friends’ of the music industry like YouTube.

You might, then, also understandably wonder if there have been significant licensing payments made by YouTube to UMG and WMG for the right to play with the AI
vocals of artists who have signed off on ‘Dream Track’ thus far. And, if so, if these payments are being placed against any as-yet-unrecouped advances that some
of these artists may have on their label accounts.
</item 52>

<item 53>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

In recent months, Warner Music Group CEO Robert Kyncl has been keeping a close eye on AI in the music industry – not surprising, given that he came to the music
biz from ‘Big Tech’, namely YouTube, where he was Chief Business Officer.

Nor is it surprising, given his current role, that he has been pushing for price hikes at music streaming services.

Both of these topics came up during Kyncl’s Q&A session at the Code Conference in Laguna Niguel, California, on Tuesday (September 26).
</item 53>

<item 54>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Furthermore, PMS is available in two options: discretionary and non-discretionary. In the former case, the portfolio managers make all decisions; in the latter
case, the investor makes all decisions.

3. Market-linked Debentures (MLD)

For HNIs looking to invest a part of their investible surplus in debt, MLDs can be a prudent option. MLDs are non-convertible debt instruments whose returns are
tied to a specific financial market benchmark, such as the Nifty 50, the BSE Sensex, or the 10-year G-Sec. The maturity periods of MLDs can vary from one year
to five years.

Though the minimum investment for MLDs is ₹1 lakh, making it easier even for small investors to invest in them, they are more popular with the HNIs owing to the
complex structure of these debentures. MLDs, unlike traditional fixed-income securities, do not provide periodic interest payments before maturity. Income
multipliers and principal protection are common features of equity-linked MLDs.
</item 54>

<item 55>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Prioritization. I think companies often use that word to mean, “we’re going to focus on one group of artists and ignore all the others.” For CMG, it means
defining what success can uniquely mean for each artist, and then delivering for them.

We identify which artists will benefit from, say, the close attention of a small group of people who will bring passion and nuance to marketing and development.
Others will clearly need the full power of our organization firing on all cylinders from the get-go.

What’s particularly changed in how you approach artist campaigns both domestically and internationally?

We engage directly with international partners earlier than ever before, and don’t unfairly rely on our ex-US affiliates within UMG to do the heavy lifting
early on. They have their own local repertoire to work on.
</item 55>

<item 56>
Psagot worth 80% less since Rani Zim bought it (Globes English | Israel Business Arena, 2023-11-05T13:10:32+00:00)
It looks sexy and glittering from the outside, but you have to understand how this sector works."

Psagot’s business has been growing, but it is not managing to translate this into profits. Its mutual funds took in NIS 1.4 billion in October, and NIS 7
billion in the year to the end of October, which means that it has NIS 43 billion under management. Most of the incoming money, however, went into money market
funds, a product with negligible management fees and low profits.

Psagot posted a loss of NIS 27 million for the first half of 2023, which is largely attributed to its non-bank credit business, an activity begun before Zim
acquired the company. About six weeks ago, Psagot sold control of its main holding in this area, AppliCheck, to its founder, Meny Guy, and now holds 20% of that
company.

So what will happen now?
</item 56>

<item 57>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Jubelirer and co. have been putting both to good use ever since.

Two years on, Jubelirer’s CMG hasn’t only returned to calmer waters – with US market share YTD in the high-5% region – it’s also achieved some glittering
successes while priming a promising array of new talent.

A key example: CMG, along with its colleagues at Capitol UK and the artist’s team, has been behind the Grammy-winning global smash that is Sam Smith’s Unholy
(an unapologetically ribald banger from a singer who refuses to be pigeonholed as a sweet balladeer).

CMG has also been a crucial partner, alongside Elliot Grainge‘s 10K Projects, in the rise of Ice Spice.
</item 57>

<item 58>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Here are a few pointers.

One, after comparing stocks based on relative valuation, you can do more layers of comparison by assessing how their valuation fares relative to growth
prospects. This apart, many other factors can justify different valuations. For example, TCS has always commanded a valuation premium to peer stocks like
Infosys, Wipro and HCLTech because of its superior margins and much more consistent growth. So just because the other stocks are cheaper on PE basis does not
mean they are cheap compared to TCS. So you can also do one more layer of comparison by assessing each company’s valuation against its long-term (5-10 years)
mean and median valuations and assess how they fare relative to their own history.

Two, equity valuations do not function in a silo. Investors have many options — fixed deposits, bonds, real estate, etc. When other assets get more attractive
one must assess scope for valuation change in equities as well.
</item 58>

<item 59>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
That price, according to HSF’s board, represented a 14.2% discount on Citrin Cooperman’s valuation of this portfolio as of September 30 (something we’ll come
back to).

As part of its announcement this morning, the HSF board took something of a swipe at its investment adviser, Merck Mercuriadis‘ Hipgnosis Song Management.

After seeing Citrin Cooperman’s latest valuation and believing it to be too high, said HSF, it then “sought advice from Hipgnosis Song Management… which is
majority owned by funds managed and/or advised by Blackstone, on their opinion on the independent valuer’s valuation.”

HSF’s board added: “Hipgnosis Song Management… eventually provided an opinion, which was heavily caveated, such that the Board has concerns as to the valuation
of the Company’s assets in its interim results.”
</item 59>

<item 60>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Even so: one wonders if some new changes at HSF, announced today, are related to this ‘continuation vote’ – and whether or not they amount to concessions to
help keep HSF investors happy and confident in the firm’s long-term prospects… and voting ‘yes, let’s keep going’.

The shareholder frustration: NAV vs. public market

First, a bit of important context.

The fundamental revenue performance of HSF doesn’t seem to be under scrutiny from shareholders: the company posted its best-ever revenues in the 12 months to
end of March, while streaming ‘pro-forma’ revenue was up 14.8% YoY in calendar 2022.

Where HSF shareholders – as well as its management – are finding frustration is in the current gulf between the firm’s ‘operative Net Asset Value’ (‘operative
NAV’) and its value on the public market (i.e. its share price).
</item 60>

<item 61>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
While Indian interest rates were always higher, investors in equities benefited from the global investors’ hunger for yields. That advantage is diminishing
today. This is not a recommendation on the stocks, but pointers on factors to consider to make your relative valuation process more foolproof.

With discipline and experience, the process will keep getting better.
</item 61>

<item 62>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Portfolio Management Services (PMS)

PMS guidelines in India date back to 1993, but this investment avenue has grown in popularity with HNIs recently. The growing financialisation of savings, the
formalisation of the economy, and the growing prosperity of tier 2 and 3 cities have resulted in an increase in AUM (excluding EPFO/PF), which stood at ₹5.28
lakh crore as of July 2023. Because the minimum investment is ₹50 lakh, this option is also only available to HNIs. Both PMS and mutual funds provide expert
guidance in navigating volatile markets and seizing investment opportunities that help create wealth. Because mutual funds cater to retail investors, their
guidelines are relatively stringent, and compliance is rigorous.

Regarding PMS, the investment guidelines are somewhat more flexible, allowing portfolio managers to outperform mutual funds. Another feature that appeals to
HNIs is that the beneficial ownership of securities is that of the investors, as opposed to mutual funds, where the ownership is that of the fund.
</item 62>

<item 63>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“Human authorship is a bedrock requirement of copyright.”

2. AI developers should be required to license materials and keep records of the materials they use for training

Many key players in the music industry and beyond have been adamant that using copyrighted works to train AI algorithms without authorization is a copyright
infringement, and a number of lawsuits have been launched against AI companies on these grounds.

One such lawsuit was filed by Universal Music Group, Concord Music Group and ABKCO against AI developer Anthropic, seeking potentially tens of millions of
dollars for the alleged “systematic and widespread infringement of their copyrighted song lyrics.”

But that doesn’t mean key music industry players don’t want their materials ever used to train AI – they just want AI developers to license that music, not
unlike how streaming services, bars and restaurants or movie producers license music.
</item 63>

<item 64>
Psagot worth 80% less since Rani Zim bought it (Globes English | Israel Business Arena, 2023-11-05T13:10:32+00:00)
Psagot doesn’t specialize in long-term savings, in which most of the money is invested outside Israel, but mainly in investment in the local market, in products
such as portfolio management, mutual funds, and a little brokerage. These are investment channels that are very vulnerable when Israel takes a hit. Finally, the
war that broke out on October 7 was a blow to the finance industry, certainly in the short term."

Now, Group Psagot is setting out to raise NIS 220 million debt, to be secured by the company’s shares in subsidiary Psagot Investment House. To that end, the
company’s management has held meetings in the past few days with several financial institutions, to present to them the investment house’s activities and
business plans. The sum that Group Psagot seeks to raise is intended both to cover the balance of its debt to Mizrahi Tefahot Bank, and to replace NIS 120
million in marketable bonds due for redemption in the coming eighteen months.
</item 64>

<item 65>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
HSF also confirmed a number of other new measures in a ‘circular’ to its shareholders today.

These measures included the fact that, if Hipgnosis Songs Fund shareholders vote to remain a publicly-traded fund (via that 2023 ‘continuation vote’) at the
firm’s next AGM, then another ‘continuation vote’ will be put in front of HSF shareholders at an Extraordinary General Meeting in January 2026.

(The ‘continuation vote’ that will take place in Q4 2023 has been scheduled ever since HSF went public in 2018 – i.e. five years after it floated. The 2026
‘Continuation Vote’, as newly-agreed, means another one is now being scheduled for around two years’ time.)

In addition, HSF today confirmed some new amendments to its ‘Investment Advisory Agreement’ (IAA) with HSM.
</item 65>

<item 66>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
At that time, he said, many predicted ad agencies were headed for the scrap-heap thanks to the arrival of Google’s AdWords, which allowed any business to set up
a customized ad campaign on Google services.

Yet ad agencies didn’t go extinct. Today, “they’re the largest customers of companies like Google,” Kyncl told the audience at the Code Conference.

“In the year 2000, when AdWords came out, everybody was predicting their deaths because all the brands [could] just go direct to AdWords and buy ads… However,
what happened was lots of different platforms emerged. Lots of ad tech emerged, complexity increased exponentially, and brands needed help with that, and the
agencies provided that value.

“And I see a very close analogy [between that and] our business. Music is incredibly broadly distributed. Everybody on earth listens to music. We’re on every
platform. And the complexity is high.
</item 66>

<item 67>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Enterprise value multiples

Enterprise value refers to total value of a company/business. This total value as perceived by the market is represented by the sum of its market cap + net debt
(includes minority interest). What works in the case of EV is that it helps in doing a capital-agnostic analysis of a company. Whether a company is funded only
with equity, or a combination of equity and debt, the total value of the business is the same. Sources of funding don’t matter.

Using EV, the value of equity (share price) is derived from the value of the business. Since the capital structure of companies varies significantly, an EV
approach enables valuing them on similar terms. One can arrive at the value of the business first and then subtract net debt, to arrive at the equity value. The
advantage with EV-based metrics is that you have more options to value a company.

Bear in mind, valuation accuracy is better the lower down the P&L and cash flow statement one can get to.
</item 67>

<item 68>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
A Final thought…

As mentioned in the opening sentence of this article, there is plenty of drama surrounding HSF right now – from the ‘discontinuation vote’ we saw at the company
in October, through to a recent lawsuit launched against Hipgnosis Songs Fund by a former business partner of Merck Mercuriadis.

Yet, amongst all of the noise, what will likely most interest HSF’s investors (including the likes of Investec Wealth & Investment, Aviva Investors, and
BlackRock) is the ongoing underlying commercial performance of the firm’s copyrights.

How are Hipgnosis Songs Fund’s assets performing vs. the market? How is the company’s ‘premium-only’ acquisition strategy holding up against its promise of
growth “uncorrelated” to other industries? And what does all of this suggest about the ongoing growth potential of the HSF portfolio vs. the multiples that
Merck Mercuriadis paid to acquire it?
</item 68>

<item 69>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
The reason for the delay? According to UK-listed HSF, it’s all to do with a recently-updated company valuation from HSF’s official independent valuer,
US-headquartered Citrin Cooperman (CC).

The HSF board says this valuation from CC was “materially higher than the valuation implied by proposed and recent transactions in the sector”.

HSF’s board particularly points to two transactions/would-be transactions as evidence:

The proposed sale earlier this year of assets to Blackstone-backed Hipgnosis Songs Capital for USD $440 million (pre-costs) – or $417.5 million (post-costs) – a
proposal that was ultimately rejected by HSF shareholders; The sale of 20,000 “non-core” songs from HSF’s portfolio to an unnamed buyer (possibly Kobalt Music
Group) for $23.1 million, as announced on December 11.
</item 69>

<item 70>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
We were so fortunate to have the talent, and friendship, of [now-Virgin Music Group President] Jacqueline [Saturn] and her team as part of our company for many
years.

“We were so successful in building [Caroline/Virgin Music], that it really was only a matter of time before the other labels within UMG would want to grow a
similar business within their groups.”

We were so successful in building that end of the business, that it really was only a matter of time before the other labels within UMG would want to grow a
similar business within their groups. It wouldn’t have made sense to have multiple indie label/artist services divisions within UMG, so the plan to globalize
Virgin and scale it to work with all the labels made so much sense.

Do I miss having Jacqueline and her team only a few steps away? Of course.
</item 70>

<item 71>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Hence, they tend to rise and fall lower than their peers. Investors with stock and mutual fund portfolios can diversify with hedge funds to reduce volatility.

e) Performing credit funds: Many operating companies cannot raise funds from banks and NBFCs due to their size or credit ratings. This is where performing
credit funds come into play. These funds specialise in credit risk assessment and, after thorough due diligence, offer credit to such companies at relatively
higher rates, resulting in higher overall returns than AAA/AA-rated bonds. These funds are attractive for investors seeking higher returns on their fixed-income
portfolio, which includes debt funds and AA/AAA-rated bonds.

f) Infrastructure Funds: These funds invest in long-term infrastructure assets owned by PSUs or private companies, such as roads, power transmission lines, and
solar power plants. Most infrastructure projects provide essential societal services and thus have intrinsic value. These assets generate cashflows in the form
of annuities backed by Central and State governments, lowering the risk of future cash flows significantly.
</item 71>

<item 72>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
India’s remarkable economic growth has resulted in a notable increase in the country’s high-net-worth investors (HNIs). According to recent wealth reports, the
population of high-net-worth individuals (HNIs) with assets worth $1 million or more, which was 7.9 lakh in 2022, will rise to 16.5 lakh, a 107 per cent
increase in five years.

The growing number of HNIs, as well as their desire to diversify their portfolios and access opportunities outside of traditional avenues such as stocks, mutual
funds, and fixed deposits, has prompted the innovation of existing investment products, leading to the emergence of a new asset class known as ‘Alternatives.’

Although the same principles of financial and investment planning apply to all investors, HNIs have an advantage. They can choose from a broader range of
conventional and unconventional investment opportunities. Investing in unconventional avenues, such as Alternatives, can improve portfolio performance, generate
alpha, and allow investors to grow their wealth faster.

Let us look at a few examples of such investment opportunities:

1.
</item 72>

<item 73>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
We’re on every platform. And the complexity is high. And the more people can upload content, and the more people can be heard, the greater the noise, which
means it’s harder to cut through the noise and sustain a career.”

But a label’s value is not just about building an artist’s career – it’s about sustaining it once that career is established as well, Kyncl explained.

“When they become established, it is also hard to remain at the top,” Kyncl said. “So their goals change. And again, we’re there to help with that.”

He added later that if you’re a music artist, “you need a team. You need an army behind you…. if you want a sustainable career, with repeatability and success.”
</item 73>

<item 74>
‘Latin artists have been paving the way for deals that offer more control, transparency, and better economics.’ (Music Business Worldwide,
2023-12-07T20:32:11+00:00)
We are in a world where music discovery and consumption are now primarily digital, and artists have now understood that. They look for partners who get it too
and whose expertise will allow them to leverage digital platforms to reach new and existing audiences wherever they are.

“Artists today are also much more conscious of their career trajectory and look for better deals, with more freedom, more transparency, and more options.”

This is especially the case in Spanish-speaking markets, where music fans are amongst the most engaged audiences in the world, whether on digital platforms or
on social media. Music content is consumed with more frequency, in different formats, across multiple platforms and through different aspects related to the
track.

Beyond that, artists today are also much more conscious of their career trajectory and look for better deals, with more freedom, more transparency, and more
options. They want to keep ownership of their masters, and get better economics.
</item 74>

<item 75>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
The genie is not going back.”

But he cautioned that the technology to develop a Content ID-style system for is “not yet developed, but… people are working on that.”

{Believe CEO Denis Ladegallerie might disagree. In comments earlier this year, Ladegallerie said that the tools to recognize and flag AI-generated content are
just about ready to go, and he expects to see them implemented in the coming few quarters.)

2) The streaming audio business won’t experience the same problems currently hitting streaming video

The streaming video business is going through tough times.

Netflix experienced its first-ever decline in subscriptions last year (though it has since roared back to health); the subscriber count of Disney+ appears to be
in freefall; and talk is growing about ongoing consolidation of the many streaming video services that have come online over the past several years.

Asked if the audio streaming business is likely to go through something similar, Kyncl’s answer was unequivocal.

“No.
</item 75>

<item 76>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
But what was maybe more surprising – for a techie like Kyncl, anyway – was his rational, level-headed explanation for why traditional music labels continue to
be relevant, and continue to thrive, in the current music ecosystem.

With the arrival of digital platforms, including open ones like YouTube and SoundCloud – and with the proliferation of indie music distribution platforms like
DistroKid and TikTok’s SoundOn – it’s inevitable that music biz observers from the world of tech like to speculating about the possibility that music labels
could, in the long-term, be headed for irrelevance.

In a world where anyone can record music using a desktop audio workstation (DAW) and upload it direct to YouTube, or to Spotify via a distribution platform,
labels have little left to do, the argument goes.

Kyncl offered strong a counterpoint to this argument at Code Media, drawing a parallel between music labels today and the advertising industry a decade or two
ago.
</item 76>

<item 77>
‘Latin artists have been paving the way for deals that offer more control, transparency, and better economics.’ (Music Business Worldwide,
2023-12-07T20:32:11+00:00)
They want to keep ownership of their masters, and get better economics.

This level of agility is at the very core of Believe’s structuring: from self-releasing services with TuneCore, to more premium services with Believe Label and
Artist Solutions or Artist Services Solutions, we pride ourselves for being able to accompany talents like Iñigo at all stages of their development, from being
emerging acts, all the way to becoming top charting artists.

Can we read anything into the fact that Inigo is Spanish, and how the trend identified above (i.e. Latin American artists having hits without being on major
record company deals) is starting to spread to different corners of the world and different genres? We just had RAYE enjoying gobal success independently with
Escapism, and in the US earlier this year there was of course Oliver Anthony going to No.1 on the Billboard Hot 100…

For many years now, Latin artists have been paving the way for deals that bring them more control, transparency, and better economics.
</item 77>

<item 78>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Depending on specific cases, you can use further variations to pick stocks. For example, during the 2003-2007 bull market, many textile stocks, for instance,
Bombay Dyeing, turned out to be multibaggers. Was their textile business booming? Not really. But the stocks were zooming because there was real estate boom and
many textile mills had manufacturing facilities in prime real estate locations. So the markets started to factor value of the company based on market value of
their land. In such a case you can deviate from Price/Book multiple to price/market value of net assets (assets – net debt).

Similar is the case with holding companies. If you think there is a reasonable case for underlying value of assets to be unlocked and Holdco discount to shrink,
like it has happened in bank holding companies in the last 2-3 years, then there is a case to move beyond book value to market value.

Enterprise value multiples

Enterprise value refers to total value of a company/business.
</item 78>

<item 79>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Do I miss having Jacqueline and her team only a few steps away? Of course. But I’m proud of the crucial roles they’ve played in Virgin’s growth and evolution,
and we work as closely together as we ever did.

On the flip side, Motown has returned as an imprint within CMG after a period as a standalone label. Obviously, that came with an inevitable downsizing process.
what shape is Motown in now and what does it add to the CMG makeup?

Motown is once again fully unified within CMG’s culture after being separate from the company for a couple of years. We have strengthened our longtime
partnership with Quality Control, with whom we continue to share tremendous success and plan for a big future.
</item 79>

<item 80>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
if you want a sustainable career, with repeatability and success.”

Here are three other things we learned from Kyncl’s talk at the Code Conference…

Credit: Tatiana Shepeleva / Shutterstock 1) Music will feel the impact of AI before other industries do – ‘within the next year’

Kyncl predicted that the music business will be out ahead of other industries when it comes to exploiting the potential of AI, and it will also – out of
necessity– be ahead on the issue of how to regulate and monetize the use of AI by non-rights-holders.

“Music – because it’s so broadly distributed and it’s so well aligned with the internet, because it’s short format, it lends itself to recommendations, it’s on
all platforms – is generally first in most transformations and most innovations. So it digitizes first,” he said.

“Whatever happened to the music industry 20 years ago is starting to happen in the movies and TV shows now.
</item 80>

<item 81>
Is Sam Bankman-Fried’s defense even trying to win? (The Verge, 2023-10-04T23:02:14+00:00)
It is a thankless task to cross-examine a customer whose money is gone

In November 2022, things went bad for Julliard. He followed Bankman-Fried on Twitter, and read aloud the “FTX is fine. Assets are fine” tweets, along with “FTX
has enough to cover all client holdings. We don’t invest client assets” and a few others, which gave Julliard the impression that his money was there — the
problem might have been technical (anti-spam measures) or regulatory. When he tried to get his money out on November 8th, it was too late. We saw screenshots of
his withdrawal attempts: $20,000 USD and about 4 Bitcoin, which were worth about $20,000 at the time: about $100,000 money, inaccessible.

It is a thankless task to cross-examine a customer whose money is gone, but Cohen tried anyway.
</item 81>

<item 82>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Before Rogier launched MasterClass, he worked as a VC, and — through his connections — he received a $500,000 seed round before he even had an idea for a
company.

And on Chain Reaction, Jacquelyn interviewed David Pakman, managing partner and head of venture investments at CoinFund. Before CoinFund, David spent 14 years
at the venture capital firm Venrock. He also led the Series A and B rounds at Dollar Shave Club, which was acquired by Unilever for $1 billion. And, in 1991,
David co-created Apple Music when he was part of Apple’s system software product marketing group.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
</item 82>

<item 83>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
Robert Kyncl, Warner Music Group

In Kyncl’s view, the solutions put in place to address UGC form a “blueprint” than can be used to address AI-generated content on platforms as well.

“When YouTube was formed… people started to upload content, including copyrighted material, which, obviously, put YouTube into hot water with lots of different
copyright holders. And I had the privilege of working through a lot of that and fix it up.

“But we made we made a very important decision, which was to go above and beyond the law, and build a fingerprinting software that allowed us to track the
copyright on our platform, and then have commercial relationship[s] with copyright holders to send them the money. Out of that we built a multi-billion-dollar
business, which now is a multi-billion-dollar business per year. And it was an incredible new revenue stream for everyone. AI is that with new super tools.”
</item 83>

<item 84>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
In response, Hipgnosis Song Management issued its own statement today: “Hipgnosis Song Management has fulfilled its duties to [HSF] with respect to both the
independent valuation and preparation of the interim results in a timely and efficient manner.

“Notwithstanding the Board’s decision to delay publication of the interim financial statements, [we] will continue to work in a constructive manner to support
the interests of the Company and its shareholders.”

The oddest thing about all of this? Citrin Cooperman’s valuation of Hipgnosis Songs Fund is, according to HSF’s own company documentation, appointed and
overseen… by the HSF board.

See this line, in every annual and interim report from HSF so far (bolding MBW’s own): “Portfolio Independent Valuer… Citrin Cooperman Advisors LLC, formerly
Massarsky Consulting, Inc., [is] appointed by the Board to independently value the Company’s Catalogues within the Portfolio.”
</item 84>

<item 85>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
That potential transaction is being managed for both HSF and HSC by two separate teams within their mutual investment advisor, Hipgnosis Song Management (HSM),
which is run and part-owned by Merck Mercuriadis.

When it announced its intention to explore the $440 million catalog sale to HSC two weeks ago, Hipgnosis Songs Fund confirmed that one of its motivating factors
was that the transaction could potentially act “as a catalyst for a re-rating of [HSF’s] share price”.

Aka: Increasing that share price to bring it up closer to the current operative NAV of the company.

HSC’s $440 million offer to acquire the 29 catalogs, HSF confirmed earlier this month, equates to a 26% increase on the amount HSF paid for those catalogs to
acquire them in the past.
</item 85>

<item 86>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Christopher Brown, JP Morgan, on the HSF board statement today (December 19)

In addition, Brown expressed his surprise that HSF asked HSM to “opine on the valuation given [HSM’s] conflict of interest”.

His point: If HSM turned around and said, “yep, HSF’s valuation is 25% too high, better bring it down”, then it would produce a new valuation for HSF that may
allow Blackstone (or the Blackstone-controlled Hipgnosis Song Capital) to launch an acquisition bid for the HSF portfolio. (Or, at least, a part of that
portfolio.)

The likely price for this hypothetical Blackstone bid would be smaller than the failed $440 million ($418.5 million post-costs) bid that HSC (via HSM) made for
a chunk of the HSF portfolio earlier this year.
</item 86>

<item 87>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
So there’s lots of growth there. And then there is the price elasticity optimization in mature markets.”

That last point is a reference to the price hikes seen among all the major music streaming services over the past year or so, and the growing belief within the
music industry that streaming price hikes will be sustained – as evidenced recently by Deezer’s second price hike in a year.

Credit: QuiteSimplyStock/Shutterstock 3) The hot music rights acquisitions space is ‘likely slowing down a bit’

The second half of 2022 saw a notable slowdown in music rights acquisitions, a phenomenon that some blamed on higher interest rates and the reduced liquidity
that came with them.

Although there has been something of a rebound this year, in Kyncl’s view, there is still a slowdown in this space – though that might be good for Warner Music
Group.

“I think what happened was there was an opening where artists and songwriters were suddenly open to selling catalogs.
</item 87>

<item 88>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
The stock would have been easily missed if PE was the metric used to spot the performers.

Hence EV-based metrics are options to consider as well when equity-based valuation multiples do not give a clear picture. Where even EBITDA multiples are not
clear, as was the case with many new-age company IPOs that were reporting losses even at the EBITDA level, investors have no option but to go further up the P&L
to metrics like EV/revenue, or even outside of P&L to use metrics like EV/subscriber.

As we mentioned earlier, relative valuation represents a number backed by a credible and well-validated story. Hence, you can use non-P&L multiples. So when you
compare companies and invest in one based on EV/Subscriber, there must be a credible story backing future monetisation of the subscriber in a profitable manner.
But do bear in mind, the risks are higher and so may be the rewards or pain as the case may be. When the risks are higher, you must apply a higher margin of
safety before choosing the stock.
</item 88>

<item 89>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
This means the net profit of A is ₹10, while that of B is ₹9. Thus although both companies are similar in terms of Price/Revenue and Price/EBITDA, A is cheaper
on the PE ratio – ie PE of A is 10 and PE of B is 11.1. On a relative valuation basis, you must choose A instead of B, but you would not have spotted it if you
had used incorrect metrics.

Hence it’s very important to use the right multiples. When it comes to equity metrics, it always has to be price or market cap (or variations of the same) in
numerator divided by line items attributable to equity holders (or variations of the same)

The accompanying table gives the commonly used equity multiples for relative valuation.

Once the metrics and importance are understood, there are, as such, no limits to how you can use them to pick stocks. You can use one or two metrics or a
combination of many more metrics to assess which stocks are undervalued or overvalued.
</item 89>

<item 90>
Michelle Jubelirer on Ice Spice, The Beatles, and reinventing Capitol Music Group: ‘I don’t think any label group has been able to turn things around as quickly
as my team and I have done in less than two years.’ (Music Business Worldwide, 2023-12-07T20:57:51+00:00)
Today, literally every plan at CMG is tailored for a specific artist or project — not pulled from a well-worn playbook — and our focus is on finding and
creating opportunities that will have an impact, no matter where an artist is in their development. The artist’s vision still lies at the heart of every plan,
but we really listen to the audience, which is smarter and savvier than ever, to determine if our marketing direction is working and if our artists are properly
engaged with their audience.

We strive to be ideas-driven and have honest constructive conversations with our artists and each other. We run our business efficiently, and are constantly
thinking about where we find real value for our artists.

Were there any other things at the top of your ‘to-do’ list?

Prioritization.
</item 90>

<item 91>
Psagot worth 80% less since Rani Zim bought it (Globes English | Israel Business Arena, 2023-11-05T13:10:32+00:00)
Psagot recently completed the sale of one third of its mutual funds business and its entire portfolio management business to Kessem, of The Phoenix Holdings
group, for NIS 200 million. The sale of the mutual funds will yield a gain to Psagot of NIS 90 million, which will be posted in it fourth quarter financials.
That amount is equivalent to the entire company’s current market cap, which, if you will, means that the rest of its business is valued at zero.

"There’s a considerable difference between commercial centers and the world of finance," a capital market player says. "People think that if they are good
managers, they can manage anything, but that isn’t necessarily so. To succeed in the capital market you have to be able to bring in the best investment managers
and create DNA in the organization. There are many elements that are not similar to real estate or retailing, and the mere fact that a person has financial
insight doesn’t mean success on the capital market.
</item 91>

<item 92>
Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI (Music Business Worldwide,
2023-11-30T21:28:03+00:00)
As reported by MBW last week, Universal Music Group submitted its response to the USCO. Unsurprisingly, amongst the highlights of UMG’s submission was the
rejection of the notion that using copyrighted material to train LLMs is “fair use”.

In his speech on Wednesday, Kooker commented directly on the flurry of submissions to the USCO from the tech industry at the end of October.

In what may have been a nod to Google’s filing and “fair use” position, Kooker told Senators that “based on recent Copyright Office filings it is clear that the
technology industry and speculative financial investors would like governments to believe in a very distorted view of copyright”.

He added: “[That view is] one in which music is considered fair use for training purposes and in which certain companies are permitted to appropriate the entire
value produced by the creative sector without permission, and to build huge businesses based on it without paying anything to the creators concerned.”
</item 92>

<item 93>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Hipgnosis Songs Fund (HSF), the UK-listed music rights fund, looks likely to hold its all-important 2023 ‘continuation vote’ on Thursday, October 26, according
to an announcement from the firm today.

Why’s that ‘likely’ rather than ‘definitely’? Because there’s still a possibility it might get bumped into November – the reasons for which we’ll get into later
in this article.

The ‘continuation vote’ will take place at HSF’s Annual General Meeting (AGM), and will see the firm’s shareholders vote on whether to keep the entity going as
a closed-ended investment entity i.e. a publicly-traded bundle of music rights.

How will that vote go? Obviously we don’t know yet – and there’s the small matter of a $440 million transaction looming that could potentially influence things.
</item 93>

<item 94>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
The NMPA made it clear it’s on the side of “opt in.”

“US copyright law is an opt-in system,” the NMPA said. “Opt-out regimes fundamentally undermine copyright protections by shifting the burden to obtain a license
away from users.”

The submission continued: “An opt-out scheme that requires rights holders to opt out on an AI company-by-AI company or application-by-application basis would
not be feasible, given the sheer volume of AI companies and applications; it is nearly a full-time job to keep up with developments in the AI marketplace…
Copyright owners, particularly individual creators and small businesses could not possibly meet such a burden.

“NMPA strongly opposes consideration of such a measure.”

“Opt-out regimes fundamentally undermine copyright protections by shifting the burden to obtain a license away from users.” NMPA

On this issue, the NMPA is fully aligned with UMG, which made a similar argument in its own submission to the Copyright Office.
</item 94>

<item 95>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
MBW Reacts is a series of analytical commentaries from Music Business Worldwide written in response to major recent entertainment events or news stories. MBW
Reacts is supported by JKBX , a technology platform that offers consumers access to music royalties as an asset class.

Legislators and regulators around the world are grappling with how to address the many issues that have emerged since AI technology “went mainstream” about a
year ago.

In the US, one of the key efforts on that front is being spearheaded by the US Copyright Office, which this past summer issued a call for submissions on the
issue of AI and copyright. The goal is for the USCO to put together a study to ”help assess whether legislative or regulatory steps in this area are warranted.”

Numerous businesses involved in AI, and those with substantial copyright holdings, have submitted their thoughts on the issue, including Universal Music Group
and AI developer Anthropic.
</item 95>

<item 96>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
“An opt-out system is based on the erroneous premise that training on copyrighted works without permission is by default lawful unless each copyright owner
objects,” UMG stated in its submission.

“That philosophy does violence to basic principles of copyright law, imposes undue burdens on copyright owners, creates the wrong incentives for AI developers,
and is neither practicable nor effective for protecting the rights of copyright owners or ensuring the sensible use of copyrighted works for training purposes.”

4. Training AI on copyrighted works is not “fair use” under almost any circumstance

Another area in which the NMPA and UMG are aligned is on the issue of whether using copyrighted material constitutes “fair use.”

Fair use is the carve-out to copyright law that allows copyrighted materials to be used without permission in certain limited circumstances, such as for
educational purposes or for reporting news.
</item 96>

<item 97>
Warner Music Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference (Music Business Worldwide,
2023-09-28T23:37:22+00:00)
“No. Music is much more resilient. So first, I think what happened in the last 15 years is incredible. They had literally zero people in the subscription model
and now we have 700 million people in the world in the premium experience… which is incredible.

“And I think really credit goes to Daniel Ek for forging the path for everybody. And then companies like Apple and YouTube and Amazon following and building up
the [music] business. It’s pretty incredible what has happened.”

“You have to embrace technology, because it’s not like you can put technology in a bottle. The genie is not going back.” Robert Kyncl, Warner Music Group

Kyncl added: “I think the opportunity ahead of us is twofold. One, the continued growth in emerging markets, and also the GDPs of those countries will be rising
at the same time. So there’s lots of growth there.
</item 97>

<item 98>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Portfolio podcast: The art of valuing stocks Portfolio podcast: The art of valuing stocks

Let there be no doubt, valuation experts like Aswath Damodaran and many successful hedge fund managers believe that absolute valuation (which we discussed in
the Big Story in bl.portfolio edition dated October 8) is the better way to arrive at the value of a stock.

However, due to a combination of complexities involved in building appropriate models, insufficient information, time constraints and also the fact that there
are hundreds of stocks you want to consider, it is just not possible to do a DCF ( discounted cash flow model) for all your stock investments, and hence
relative valuation as an approach to picking stocks too is well accepted and widely prevalent.

When you decide to buy or sell a stock based on its PE or P/B or EV/EBITDA or EV/Revenue or even EV/subscriber ratios, you are engaging in the art of relative
valuation.
</item 98>

<item 99>
Fundamental Investing: The Art of Relative Valuation (Business Line, 2023-10-14T15:38:28+00:00)
Any other metric could be wrong and misleading.

For example, price by revenue or price by EBITDA is incorrect. Why? How do you differentiate between two companies A and B — both having same market cap of ₹100
and similar in all means except for debt levels — same revenue (₹100) and same gross margin and same EBITDA margins of 20 per cent and thus same EBITDA of ₹20.
But A has zero debt, while B has debt of ₹10. Since revenue or sales and EBITDA are same, if you do a market cap/revenue or Price /Sales, or marketcap/EBITDA,
both companies will have the same valuation.

However, let’s assume depreciation is ₹10, interest cost is 10 per cent and taxes are zero. This means the net profit of A is ₹10, while that of B is ₹9.
</item 99>

<item 100>
NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’ (Music Business Worldwide, 2023-11-30T21:28:50+00:00)
That argument could yet come into play in Anthropic’s legal battle with Universal.

And given the high stakes involved – for the music industry, for the tech industry and for the shape of culture going forward – that case, along with the US
Copyright Office’s study of AI and copyright, are worth keeping a close eye on.

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 100>
=== END ITEM id=b27d40e3c2fbd070 ===
