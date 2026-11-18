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

This file has 4 items.

=== ITEM id=08c9ac02049584ac ===
Gold sentences and facts:
Sentence 0: Both PMS and mutual funds provide expert guidance in navigating volatile markets and seizing investment opportunities that help create wealth.
  Fact 0.0: PMS provides expert guidance in navigating volatile markets.
  Fact 0.1: Mutual funds provide expert guidance in navigating volatile markets.
  Fact 0.2: PMS provides expert guidance in seizing investment opportunities that help create wealth.
  Fact 0.3: Mutual funds provide expert guidance in seizing investment opportunities that help create wealth.
Sentence 1: Brown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results, especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s previous set of results (to end of March 2023).
  Fact 1.0: Brown added that he was surprised.
  Fact 1.1: Brown was surprised that a second valuation expert had not been drafted in to value the HSF portfolio for the latest interim results.
  Fact 1.2: Kroll was appointed to consider the reasonableness of Citrin Cooperman's assumptions for HSF's previous set of results.
  Fact 1.3: HSF's previous set of results runs to the end of March 2023.

Retrieved items:
<item 1>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
HSF’s investments in seasonal Christmas ‘evergreens’ continue to pay off too, with cuts on Michael Buble’s Christmas album and – most famously – a share of
Mariah Carey’s perennial megahit, All I Want For Christmas Is You.

JKBX (pronounced "Jukebox") unlocks shared value from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares.
In short: JKBX makes it possible for you to invest in music the same way you invest in stocks and other securities.Music Business Worldwide
</item 1>

<item 2>
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
</item 2>

<item 3>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
That price, according to HSF’s board, represented a 14.2% discount on Citrin Cooperman’s valuation of this portfolio as of September 30 (something we’ll come
back to).

As part of its announcement this morning, the HSF board took something of a swipe at its investment adviser, Merck Mercuriadis‘ Hipgnosis Song Management.

After seeing Citrin Cooperman’s latest valuation and believing it to be too high, said HSF, it then “sought advice from Hipgnosis Song Management… which is
majority owned by funds managed and/or advised by Blackstone, on their opinion on the independent valuer’s valuation.”

HSF’s board added: “Hipgnosis Song Management… eventually provided an opinion, which was heavily caveated, such that the Board has concerns as to the valuation
of the Company’s assets in its interim results.”
</item 3>

<item 4>
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
</item 4>

<item 5>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
“[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try and pin the
blame on [Hipgnosis Songs Management].” Christopher Brown, JP Morgan

Concluded Brown: “[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try
and pin the blame on [Hipgnosis Songs Management].

“Our view is that a breakdown in the relationship between Board and manager would not be optimal for maximising shareholder value.”

Brown further noted that JP Morgan would “not be surprised to see Citrin Cooperman resign, having been very publicly undermined by the [HSF] Board today”.

JP Morgan retained its ‘overweight’ rating against HSF’s share price “in spite of yet another [HSF] own goal”.
</item 5>

<item 6>
Investment options for high networth investors (Business Line, 2023-10-28T14:42:28+00:00)
Portfolio Management Services (PMS)

PMS guidelines in India date back to 1993, but this investment avenue has grown in popularity with HNIs recently. The growing financialisation of savings, the
formalisation of the economy, and the growing prosperity of tier 2 and 3 cities have resulted in an increase in AUM (excluding EPFO/PF), which stood at ₹5.28
lakh crore as of July 2023. Because the minimum investment is ₹50 lakh, this option is also only available to HNIs. Both PMS and mutual funds provide expert
guidance in navigating volatile markets and seizing investment opportunities that help create wealth. Because mutual funds cater to retail investors, their
guidelines are relatively stringent, and compliance is rigorous.

Regarding PMS, the investment guidelines are somewhat more flexible, allowing portfolio managers to outperform mutual funds. Another feature that appeals to
HNIs is that the beneficial ownership of securities is that of the investors, as opposed to mutual funds, where the ownership is that of the fund.
</item 6>

<item 7>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
Worth remembering: The HSF board has known about Citrin Cooperman’s latest valuation (as of September 30) for at least the past eight days.

We know this because, as mentioned, on December 11, HSF announced that the $23.1 million price of an agreed HSF ‘non-core’ asset sale represented a 14.2%
discount vs. Citrin Cooperman’s latest valuation of the portfolio (i.e. the HSF board had seen this latest valuation).

Yet it took until today (December 19) for HSF to alert its investors to its dissatisfaction with this valuation.

WHAT does JP Morgan make of it all?

One of the closest watchers of Hipgnosis Songs Fund in recent years has been JP Morgan and its analyst Christopher Brown.
</item 7>

<item 8>
Graph query row:
article: Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it?
publisher: Music Business Worldwide
connections: [{'related': 'Music Business Worldwide', 'relationship': 'PUBLISHED_BY'}, {'related': '1989 (Taylor’s Version)', 'relationship': 'MENTIONS'},
{'related': 'Taylor Swift', 'relationship': 'MENTIONS'}, {'related': 'Jack Antonoff', 'relationship': 'MENTIONS'}, {'related': 'Rob Naylor', 'relationship':
'MENTIONS'}, {'related': 'Hipgnosis Songs Fund', 'relationship': 'MENTIONS'}, {'related': 'HSF board', 'relationship': 'MENTIONS'}, {'related': 'company
valuation', 'relationship': 'MENTIONS'}, {'related': 'Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million
(post-costs)', 'relationship': 'MENTIONS'}, {'related': 'Hipgnosis Songs Capital', 'relationship': 'MENTIONS'}, {'related': 'Sale of 20,000 non-core songs from
HSF’s portfolio for $23.1 million', 'relationship': 'MENTIONS'}, {'related': 'unnamed buyer', 'relationship': 'MENTIONS'}, {'related': 'Kobalt Music Group',
'relationship': 'MENTIONS'}, {'related': 'valuation', 'relationship': 'MENTIONS'}, {'related': 'Citrin Cooperman Advisors LLC', 'relationship': 'MENTIONS'},
{'related': 'Hipgnosis Songs Fund board', 'relationship': 'MENTIONS'}, {'related': 'Christopher Mills', 'relationship': 'MENTIONS'}, {'related': 'Mercuriadis',
'relationship': 'MENTIONS'}, {'related': 'HSF’s portfolio', 'relationship': 'MENTIONS'}, {'related': 'music catalogues', 'relationship': 'MENTIONS'},
{'related': 'call option', 'relationship': 'MENTIONS'}, {'related': 'discount rate', 'relationship': 'MENTIONS'}, {'related': 'interest rate rises',
'relationship': 'MENTIONS'}, {'related': 'Hipgnosis Songs Fund’s portfolio', 'relationship': 'MENTIONS'}, {'related': 'JP Morgan', 'relationship': 'MENTIONS'},
{'related': 'Brown', 'relationship': 'MENTIONS'}, {'related': 'Citrin Cooperman', 'relationship': 'MENTIONS'}, {'related': 'Kroll', 'relationship': 'MENTIONS'},
{'related': 'current market uncertainty', 'relationship': 'MENTIONS'}, {'related': 'HSF portfolio', 'relationship': 'MENTIONS'}, {'related': 'HSF shares',
'relationship': 'MENTIONS'}, {'related': 'Net Asset Value', 'relationship': 'MENTIONS'}, {'related': 'HSM', 'relationship': 'MENTIONS'}, {'related':
'Blackstone', 'relationship': 'MENTIONS'}, {'related': 'Hipgnosis Song Capital', 'relationship': 'MENTIONS'}, {'related': 'Failed $440 million bid for a chunk
of the HSF portfolio', 'relationship': 'MENTIONS'}, {'related': 'Hypothetical Blackstone acquisition bid for the HSF portfolio', 'relationship': 'MENTIONS'},
{'related': 'conflict of interest', 'relationship': 'MENTIONS'}, {'related': 'Hipgnosis Songs Management', 'relationship': 'MENTIONS'}, {'related': 'Christopher
Brown', 'relationship': 'MENTIONS'}, {'related': 'discontinuation vote', 'relationship': 'MENTIONS'}, {'related': 'Investec Wealth & Investment',
'relationship': 'MENTIONS'}, {'related': 'Aviva Investors', 'relationship': 'MENTIONS'}, {'related': 'copyrights', 'relationship': 'MENTIONS'}, {'related':
'premium-only acquisition strategy', 'relationship': 'MENTIONS'}, {'related': 'Jack Antonoff catalog', 'relationship': 'MENTIONS'}, {'related': 'Lover',
'relationship': 'MENTIONS'}, {'related': 'Joel Little', 'relationship': 'MENTIONS'}, {'related': 'Joel Little catalog', 'relationship': 'MENTIONS'}, {'related':
'Pro-Forma Revenue', 'relationship': 'MENTIONS'}, {'related': 'Michael Buble', 'relationship': 'MENTIONS'}, {'related': 'Christmas album', 'relationship':
'MENTIONS'}, {'related': 'Mariah Carey', 'relationship': 'MENTIONS'}, {'related': 'All I Want For Christmas Is You', 'relationship': 'MENTIONS'}, {'related':
'music royalties', 'relationship': 'MENTIONS'}, {'related': 'lawsuit', 'relationship': 'MENTIONS'}, {'related': 'Royalty Shares', 'relationship': 'MENTIONS'},
{'related': 'JKBX', 'relationship': 'MENTIONS'}, {'related': 'BlackRock', 'relationship': 'MENTIONS'}, {'related': 'Hipgnosis Song Management', 'relationship':
'MENTIONS'}, {'related': 'Merck Mercuriadis', 'relationship': 'MENTIONS'}, {'related': 'music', 'relationship': 'MENTIONS'}]
passages: [{'page': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music
Business Worldwide, 2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company.
What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)\nWHAT’S HAPPENED?\n\nMBW Explains is a series of analytical features in
which we explore the context behind major music industry talking points – and suggest what might happen next. MBW Explains is supported by JKBX , a technology
platform that offers consumers access to music royalties as an asset class.\n\nThe drama at Hipgnosis Songs Fund isn’t letting up for the Holidays, it
seems.\n\nThis morning (December 19), the company’s board – now led by Chairman, Rob Naylor – issued a statement announcing that HSF would no longer, as
previously expected, be issuing its latest financial results today.\n\nInstead, the results (covering the six months to end of September 2023) are now expected
to be published sometime before the close of December 31.\n\nThe reason for the delay?'}, {'page': 'Hipgnosis Songs Fund’s board is concerned over Citrin
Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs
Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)\nThe reason for the delay? According to UK-listed HSF, it’s all to do with a recently-updated company valuation from HSF’s official
independent valuer, US-headquartered Citrin Cooperman (CC).\n\nThe HSF board says this valuation from CC was “materially higher than the valuation implied by
proposed and recent transactions in the sector”.\n\nHSF’s board particularly points to two transactions/would-be transactions as evidence:\n\nThe proposed sale
earlier this year of assets to Blackstone-backed Hipgnosis Songs Capital for USD $440 million (pre-costs) – or $417.5 million (post-costs) – a proposal that was
ultimately rejected by HSF shareholders; The sale of 20,000 “non-core” songs from HSF’s portfolio to an unnamed buyer (possibly Kobalt Music Group) for $23.1
million, as announced on December 11.'}, {'page': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it
going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s
latest valuation of the company. What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)\nThat price, according to HSF’s board,
represented a 14.2% discount on Citrin Cooperman’s valuation of this portfolio as of September 30 (something we’ll come back to).\n\nAs part of its announcement
this morning, the HSF board took something of a swipe at its investment adviser, Merck Mercuriadis‘ Hipgnosis Song Management.\n\nAfter seeing Citrin
Cooperman’s latest valuation and believing it to be too high, said HSF, it then “sought advice from Hipgnosis Song Management… which is majority owned by funds
managed and/or advised by Blackstone, on their opinion on the independent valuer’s valuation.”\n\nHSF’s board added: “Hipgnosis Song Management… eventually
provided an opinion, which was heavily caveated, such that the Board has concerns as to the valuation of the Company’s assets in its interim results.”'},
{'page': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business
Worldwide, 2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it
going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)\nIn response, Hipgnosis Song Management issued its own statement today: “Hipgnosis
Song Management has fulfilled its duties to [HSF] with respect to both the independent valuation and preparation of the interim results in a timely and
efficient manner.\n\n“Notwithstanding the Board’s decision to delay publication of the interim financial statements, [we] will continue to work in a
constructive manner to support the interests of the Company and its shareholders.”\n\nThe oddest thing about all of this? Citrin Cooperman’s valuation of
Hipgnosis Songs Fund is, according to HSF’s own company documentation, appointed and overseen… by the HSF board.\n\nSee this line, in every annual and interim
report from HSF so far (bolding MBW’s own): “Portfolio Independent Valuer… Citrin Cooperman Advisors LLC, formerly Massarsky Consulting, Inc., [is] appointed by
the Board to independently value the Company’s Catalogues within the Portfolio.”'}, {'page': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s
latest valuation of the company. What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s
board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)\nNot to mention this line in various HSF company reports to date: “The [HSF] Board is ultimately and solely responsible for
overseeing the valuation of the Company’s investments in music catalogues and has appointed [Citrin Cooperman] to perform this specialist work.”\n\nWHAT’S
really going on?\n\nIt’s a fair bet that Hipgnosis Songs Fund’s board – which last week appointed activist investor Christopher Mills as a non-exec Director –
is making something of a ‘hidden’ statement to shareholders with today’s announcement.\n\nIt may be a coded attempt by HSF’s board to gnash teeth towards
HSM/Mercuriadis’ ‘call option’ – which, as MBW has pointed out before, leaves the exec with the exclusive power to sweep in and acquire HSF’s portfolio under
several different circumstances.'}, {'page': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going
to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest
valuation of the company. What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)\nThere are a few clues in HSF’s latest public
statement to this end, perhaps including its rare – not to mention conspicuous – mention of HSM being “majority-owned by funds managed and/or advised by
Blackstone“.\n\nIn addition, today’s statement is also a clear signal from the HSF board that it has understandable concerns over the now-infamous ‘discount
rate’ used by Citrin Cooperman to value HSF’s portfolio (as MBW covered in depth through here).\n\nCitrin Cooperman continues to stick to an 8.5% discount rate
in its valuations, despite interest rate rises over the past 12-18 months – and has long refused to move this 8.5% figure upwards to, say, 9% or 9.5%.\n\nIf
Citrin Cooperman ever did raise this discount rate, the independent valuation of Hipgnosis Songs Fund’s portfolio would, in turn, instantly fall.'}, {'page':
'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to
do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)\nWorth remembering: The HSF board has known about Citrin Cooperman’s latest valuation (as of
September 30) for at least the past eight days.\n\nWe know this because, as mentioned, on December 11, HSF announced that the $23.1 million price of an agreed
HSF ‘non-core’ asset sale represented a 14.2% discount vs. Citrin Cooperman’s latest valuation of the portfolio (i.e. the HSF board had seen this latest
valuation).\n\nYet it took until today (December 19) for HSF to alert its investors to its dissatisfaction with this valuation.\n\nWHAT does JP Morgan make of
it all?\n\nOne of the closest watchers of Hipgnosis Songs Fund in recent years has been JP Morgan and its analyst Christopher Brown.'}, {'page': 'Hipgnosis
Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to
do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)\nIn a rather damning note issued today covering the delay in the publication of HSF’s
results, Brown wrote: “This is an early blow to the credibility of the new [Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the
independent valuer, Citrin Cooperman.”\n\nBrown added that he was “surprised” to see that a second valuation expert had not been drafted in to value the HSF
portfolio for the latest interim results, especially as Kroll was appointed to consider the “reasonableness” of Citrin Cooperman’s assumptions for HSF’s
previous set of results (to end of March 2023).\n\nAdded Brown: “We think the easiest solution now for the [HSF] Board is to apply an additional discount rate
to the [Citrin Cooperman] valuation to reflect current market uncertainty, on the basis that it is probably too late to employ another valuer.”'}, {'page':
'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to
do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)\nBrown noted that a 0.5% rise in Citrin Cooperman’s discount rate (i.e. raising it from 8.5%
to 9.0%) would “reduce the fair value of the [HSF] portfolio by 7.9%, which on a leveraged basis is [approximately a] 10% reduction in NAV [Net Asset
Value]”.\n\nShould HSF’s board go down this route, he noted, he expected “the [HSF] shares to weaken… given that this represents another ‘unforced
error’”.\n\n“This is an early blow to the credibility of the new [Hipgnosis Songs Fund] Board, and casts further doubt over the credibility of the independent
valuer, Citrin Cooperman.”'}, {'page': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do
about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation
of the company. What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)\nChristopher Brown, JP Morgan, on the HSF board statement
today (December 19)\n\nIn addition, Brown expressed his surprise that HSF asked HSM to “opine on the valuation given [HSM’s] conflict of interest”.\n\nHis
point: If HSM turned around and said, “yep, HSF’s valuation is 25% too high, better bring it down”, then it would produce a new valuation for HSF that may allow
Blackstone (or the Blackstone-controlled Hipgnosis Song Capital) to launch an acquisition bid for the HSF portfolio. (Or, at least, a part of that
portfolio.)\n\nThe likely price for this hypothetical Blackstone bid would be smaller than the failed $440 million ($418.5 million post-costs) bid that HSC (via
HSM) made for a chunk of the HSF portfolio earlier this year.'}, {'page': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of
the company. What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s board is concerned over
Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)\n“[It] is up to the
[Hipgnosis Songs Fund] Board to satisfy itself that the independent valuer, whom it hires, is up to the job, rather than try and pin the blame on [Hipgnosis
Songs Management].” Christopher Brown, JP Morgan\n\nConcluded Brown: “[It] is up to the [Hipgnosis Songs Fund] Board to satisfy itself that the independent
valuer, whom it hires, is up to the job, rather than try and pin the blame on [Hipgnosis Songs Management].\n\n“Our view is that a breakdown in the relationship
between Board and manager would not be optimal for maximising shareholder value.”\n\nBrown further noted that JP Morgan would “not be surprised to see Citrin
Cooperman resign, having been very publicly undermined by the [HSF] Board today”.\n\nJP Morgan retained its ‘overweight’ rating against HSF’s share price “in
spite of yet another [HSF] own goal”.'}, {'page': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it
going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s
latest valuation of the company. What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)\nA Final thought…\n\nAs mentioned in the
opening sentence of this article, there is plenty of drama surrounding HSF right now – from the ‘discontinuation vote’ we saw at the company in October, through
to a recent lawsuit launched against Hipgnosis Songs Fund by a former business partner of Merck Mercuriadis.\n\nYet, amongst all of the noise, what will likely
most interest HSF’s investors (including the likes of Investec Wealth & Investment, Aviva Investors, and BlackRock) is the ongoing underlying commercial
performance of the firm’s copyrights.\n\nHow are Hipgnosis Songs Fund’s assets performing vs. the market? How is the company’s ‘premium-only’ acquisition
strategy holding up against its promise of growth “uncorrelated” to other industries? And what does all of this suggest about the ongoing growth potential of
the HSF portfolio vs. the multiples that Merck Mercuriadis paid to acquire it?'}, {'page': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s
latest valuation of the company. What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)', 'passage': 'Hipgnosis Songs Fund’s
board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)\nthe multiples that Merck Mercuriadis paid to acquire it?\n\nSuch questions will only be answered when Hipgnosis Songs Fund’s board
elects to release the company’s six-monthly financial results to end of September 30.\n\nThe most important KPI within those results, in MBW’s view: the
company’s Pro-Forma Revenue, and how it compares, like-for-like, vs. the previous year’s equivalent period.\n\nIn the meantime, back in the ‘real world’ of the
music biz, HSF is actually enjoying quite a fanciable Q4.\n\nThe company is benefitting from its cuts on hit albums including Taylor Swift’s 1989 (Taylor’s
Version), via its acquired Jack Antonoff catalog, as well as Swift’s Lover (via its acquired Joel Little catalog).'}, {'page': 'Hipgnosis Songs Fund’s board is
concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide, 2023-12-19T23:02:01+00:00)',
'passage': 'Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business
Worldwide, 2023-12-19T23:02:01+00:00)\nHSF’s investments in seasonal Christmas ‘evergreens’ continue to pay off too, with cuts on Michael Buble’s Christmas
album and – most famously – a share of Mariah Carey’s perennial megahit, All I Want For Christmas Is You.\n\nJKBX (pronounced "Jukebox") unlocks shared value
from things people love by offering consumers access to music as an asset class — it calls them Royalty Shares. In short: JKBX makes it possible for you to
invest in music the same way you invest in stocks and other securities.Music Business Worldwide'}]
</item 8>

<item 9>
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
</item 9>

<item 10>
Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? (Music Business Worldwide,
2023-12-19T23:02:01+00:00)
the multiples that Merck Mercuriadis paid to acquire it?

Such questions will only be answered when Hipgnosis Songs Fund’s board elects to release the company’s six-monthly financial results to end of September 30.

The most important KPI within those results, in MBW’s view: the company’s Pro-Forma Revenue, and how it compares, like-for-like, vs. the previous year’s
equivalent period.

In the meantime, back in the ‘real world’ of the music biz, HSF is actually enjoying quite a fanciable Q4.

The company is benefitting from its cuts on hit albums including Taylor Swift’s 1989 (Taylor’s Version), via its acquired Jack Antonoff catalog, as well as
Swift’s Lover (via its acquired Joel Little catalog).
</item 10>
=== END ITEM id=08c9ac02049584ac ===

=== ITEM id=17692d87bfb72621 ===
Gold sentences and facts:
Sentence 0: Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).
  Fact 0.0: Bankman-Fried was compared to Warren Buffet.
  Fact 0.1: Many people called Bankman-Fried the white horse of crypto.
  Fact 0.2: TechCrunch never called Bankman-Fried the white horse of crypto.
Sentence 1: Wang testified that he heard Bankman-Fried tell another trader that it was okay to continue withdrawing money from the account as long as the amount was less than FTX’s total trading revenue.
  Fact 1.0: Wang testified that he heard Bankman-Fried tell another trader.
  Fact 1.1: Bankman-Fried told another trader it was okay to continue withdrawing money from the account.
  Fact 1.2: Bankman-Fried said the withdrawals were okay as long as the amount was less than FTX's total trading revenue.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
He could face up to 115 years in jail if convicted on all charges.

The crypto industry as a whole suffered from FTX’s collapse, which was the first of many. BlockFi filed for Chapter 11 in November 2022, as did Genesis Global
Trading in January.

Where we are today

Bankman-Fried will be represented by Cohen & Gresser, and Mark Cohen, a high-profile defense attorney and former federal prosecutor, will be the lead attorney.
If that name sounds familiar, it might be because he also represented Ghislaine Maxwell in her sex trafficking trial related to Jeffrey Epstein. He requested an
early release for Bankman-Fried but was denied.

With the trial beginning, we’re witnessing how FTX’s story ends. But what’s on our mind is what happens to the investors and creditors affected by the collapse?
And what happens to the billions in crypto assets tied up in legal proceedings?
</item 1>

<item 2>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
SBF’s trial has started, this is how he and FTX got here

The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of
seven counts of fraud and conspiracy. And as one former federal prosecutor put it: “The odds seem to be stacked against him at this point.”

The 31-year-old co-founded FTX in 2019; within a few years the once third-largest crypto exchange’s valuation hit $32 billion at its peak. It’s now trying to
claw back any funds to distribute to creditors.

But how did the once third-largest crypto exchange get here?

Before FTX, Bankman-Fried co-founded crypto-trading firm Alameda Research in 2017. He co-founded FTX in 2019 as a complement to Alameda, to help bring in
revenue and liquidity for the trading arm.
</item 2>

<item 3>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
In a published DM exchange with a Vox reporter, he said he regretted filing for bankruptcy and thought that “regulators make everything worse.”

SBF arrested

Bankman-Fried was arrested in December 2022 in the Bahamas, where FTX was based. He was then extradited to the U.S. to face a number of criminal charges. He was
released on a $250 million bail bond, and he remained under house arrest at his parents’ home in Palo Alto. This was revoked in August after he was accused of
intimidating Alameda’s former CEO, Caroline Ellison, by leaking her private diary.

Ray represented the company during a House Financial Services Committee hearing regarding FTX. When asked whether the firm had significant risk management
systems, Ray said at the time that “there were virtually no internal controls and no separateness whatsoever” and added that he did not “trust a single piece of
paper” in the exchange’s organization. U.S.
</item 3>

<item 4>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
SBF trial: Everything to know from the FTX courtroom ahead of his testimony Sam Bankman-Fried is in court for alleged fraud and money laundering while leading
crypto exchange FTX and sister company Alameda. Here’s our latest coverage as the case nears its end.

The trial of Bankman-Fried, the former CEO and co-founder of collapsed crypto exchange FTX, is entering its next chapter as proceedings resume Thursday. As
confirmed during a teleconference Wednesday, SBF will be taking the stand, along with a handful of other witnesses as the onus shifts toward the defense after
the prosecution brought its case to its conclusion.

As former federal prosecutor Josh Naftalis told TechCrunch, SBF taking to the stand in his own trial is a “Hail Mary” ahead of a verdict and potential
sentencing for seven counts tied to fraud and money laundering.

“Once it goes to cross[-examination], he doesn’t get to say, ‘I’m done,’” Naftalis said.
</item 4>

<item 5>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Then, Nishad Singh, about a month after founding Alameda Research. Finally, Caroline Ellison.

Not having a risk team, when you are any kind of financial anything, is certainly a choice

Though Bankman-Fried was the CEO, and also the majority owner, he wanted to be clear: he did not supervise Wang’s direct work. Anyway, after a bunch of wildly
successful arbitrage — 50 percent to 100 percent annualized returns, per his testimony — he decided to found a cryptocurrency exchange, FTX. He figured he’d
fail; that there was only a 20 percent chance of success. Bankman-Fried did not define what he expected the time period to be on this estimate, but arguably 20
percent was a much higher chance of success than FTX would enjoy once Alameda dipped into the customer deposits.

By the way, because he was such a good guy, Bankman-Fried made a point of “periodically” handling support tickets himself.
</item 5>

<item 6>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 6>

<item 7>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.
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
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
That means it purported to have more FTT tokens on its balance sheet than what existed in the world.

Around the same time it was exposed, the world’s largest crypto exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and
FTT. (It had an equity position in FTX from 2019 to 2021.) This essentially triggered a bank run on FTX.

FTX and Alameda filed for Chapter 11 bankruptcy in the U.S. mid-November 2022. Bankman-Fried resigned, and John J. Ray III, the Enron turnaround veteran, was
appointed its new CEO.

Bankman-Fried, however, maintained his innocence. At The New York Times’ DealBook Summit, he appeared virtually from the Bahamas, saying “I didn’t ever try to
commit fraud on anyone; I was shocked by what happened this month.”
</item 9>

<item 10>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
regulators and government officials, many of whom he donated to.

Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).

But in early November 2022, that all changed.

FTX’s collapse

Concerns surrounding FTX’s liquidity grew after CoinDesk published a copy of Alameda’s balance sheet, showing the firm held $14.6 billion in assets and $8
billion in liabilities as of June 30, 2022.

But there was a problem: The report showed Alameda’s largest asset was $3.66 billion of “unlocked FTT” and $2.16 billion of “FTT collateral.” FTT was the token
behind FTX.

The balance sheet showed that the $5.82 billion in FTT tokens that Alameda owned was 193% higher than the total FTT market cap, which was about $3 billion at
the time.
</item 10>
=== END ITEM id=17692d87bfb72621 ===

=== ITEM id=e46786431adffd88 ===
Gold sentences and facts:
Sentence 0: Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
  Fact 0.0: Google unveiled new hardware.
  Fact 0.1: This week was Google's annual hardware event.
  Fact 0.2: At the event, Google showed off what it has been working on.
  Fact 0.3: Google is a search and consumer tech giant.
Sentence 1: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 1.0: Wired removed the article for not meeting its standards.
  Fact 1.1: The article was widely shared on social media.
  Fact 1.2: The article was boosted in write-ups at other outlets.
  Fact 1.3: The degree to which the article was shared and boosted shows how willing people are to accept foul play on Google's part.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content, their readers and ad revenue.
  Fact 2.2: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 1>

<item 2>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 4>

<item 5>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 5>

<item 6>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 7>

<item 8>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 10>
=== END ITEM id=e46786431adffd88 ===

=== ITEM id=b9e0a9c3c124ec81 ===
Gold sentences and facts:
Sentence 0: Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense, especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.
  Fact 0.0: Todd Boehly is American.
  Fact 0.1: Todd Boehly is a co-owner of Chelsea.
  Fact 0.2: Expanding Chelsea's profile and reach in the U.S. would make sense under Todd Boehly's co-ownership.
  Fact 0.3: Catarina Macario is a USWNT international.
  Fact 0.4: Mia Fishel is a USWNT international.
  Fact 0.5: Catarina Macario and Mia Fishel play their club football in the U.S.
  Fact 0.6: CBS Sports holds WSL rights.
Sentence 1: It was 24 September, Chelsea’s second home Premier League defeat on the spin after the one against Nottingham Forest and they were booed off again.
  Fact 1.0: The match was on 24 September.
  Fact 1.1: The match was Chelsea's second home Premier League defeat in a row.
  Fact 1.2: The defeat followed the one against Nottingham Forest.
  Fact 1.3: Chelsea were booed off again.
Sentence 2: Late checks will be made on the fitness of Reece James, Axel Disasi and Nicolas Jackson, while Benoit Badiashile could return from a long-term injury.
  Fact 2.0: Late checks will be made on the fitness of Reece James.
  Fact 2.1: Late checks will be made on the fitness of Axel Disasi.
  Fact 2.2: Late checks will be made on the fitness of Nicolas Jackson.
  Fact 2.3: Benoit Badiashile could return from a long-term injury.

Retrieved items:
<item 1>
Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’ (The Independent - Sports, 2023-11-15T22:32:05+00:00)
That is an understatement. And if there seemed a risk involved in swapping the stability of City for the seeming madhouse of Stamford Bridge, there was evidence
of more cool-headed thinking from a 21-year-old. Chelsea have been manic buyers but he said: “I did look at how many players were there but I don’t think people
realise how many players they actually got rid off, as well.”

The departures of, among others, Mason Mount, Kai Havertz, Christian Pulisic and Joao Felix, meant there could be opportunities in the attacking midfield
positions. “I looked at the squad and thought: ‘If I go there and know what I can do then I’ll have the chance of playing’,” he said. “And thankfully it’s
happened.” If there seemed plenty of traffic in the queue for places in Southgate’s attack, Palmer seems to have accelerated past several rivals. Now an England
debut could follow in the next week.
</item 1>

<item 2>
Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn (The Guardian, 2023-11-01T22:03:18+00:00)
For more information see our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our
Privacy Policy . We use Google reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

But the Swedish goalkeeper was at fault when Chelsea finally found the goal they were looking for, palming Gallagher’s cross after he had combined with Palmer
straight to the feet of Badiashile and he calmly sidefooted home.

Blackburn’s supporters then felt aggrieved when a Callum Brittain cross struck Gallagher’s trailing arm but once again Robinson waved away strong appeals for a
penalty.

The visitors showed they had no intention of surrendering without a fight when Harry Leonard caught Axel Disasi napping and raced onto a long ball at the start
of the second half, only to curl his shot wide of the post. Not for the first time, Robert Sánchez was then far too casual as the Chelsea goalkeeper’s pass was
almost intercepted.
</item 2>

<item 3>
Arteta reveals Pochettino influence on his career | Poch: He is family (Sky Sports, 2023-10-20T21:30:00+00:00)
Havertz will command particular focus after his early struggles at Arsenal. With just one goal in 12 appearances for the club there are questions about his form
and role.

Please use Chrome browser for a more accessible video player Ahead of Kai Havertz's return to Stamford Bridge, Sam Blitz takes a forensic look at his tactical
role at Arsenal

Armando Broja will miss Chelsea's game against Arsenal due to "irritation" in his knee. Late checks will be made on the fitness of Reece James, Axel Disasi and
Nicolas Jackson, while Benoit Badiashile could return from a long-term injury.

Enzo Fernandez and Moises Caicedo should be available despite travelling to South America during the international break.

Ben Chilwell, Carney Chukwuemeka, Christopher Nkunku, Romeo Lavia, Wesley Fofana and Trevoh Chalobah remain unavailable.

Arsenal will make a late call on the fitness of Bukayo Saka, William Saliba and Leandro Trossard.
</item 3>

<item 4>
Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn (The Guardian, 2023-11-01T22:03:18+00:00)
“But we need to be careful and little by little he will get back to full fitness.”

Blackburn’s fortunes since Andy Cole scored the winning goal that saw off Tottenham in the final of this competition more than 20 years ago under Graeme Souness
have seen the Lancashire club drop down to the third tier for a season before coming close to reaching the Championship playoffs in the past two campaigns.

They currently find themselves 12th after an inconsistent start but Tomasson opted against sending out his first-choice team despite Chelsea’s recent
vulnerabilities at home, with the 26-year-old captain Lewis Travis the oldest player he selected.

Benoît Badiashile opens the scoring for Chelsea from close range in the first half. Photograph: Javier García/Shutterstock

“It was the millionaires of the Premier League against the youngsters of the Championship,” reflected the former Denmark forward. “We should be proud of this
performance.”
</item 4>

<item 5>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
Soccer’s planned combined headquarters and national training facility in Georgia. As of yet, there’s no targeted date set for the move.

Hayes, too, will benefit in ways other than compensation and prestige. She spoke to reporters about looking forward to spending more time with her five-year-old
son, Harry. She has never been to one of his sports days, picked him up from school or taken him to an after-school club and she wants to do that.

The main visible wrinkle in the process was Chelsea’s surprise move of announcing Hayes’ departure on Saturday, Nov. 4. With the contract not yet finalized and
U.S. Soccer board approval still needed, Chelsea issued their statement at 11 a.m. ET in the U.S. in which it noted she would leave at the end of the season “to
pursue a new opportunity outside of the WSL and club football.” The club feared that the news was starting to leak and wanted to share the news on its own.
</item 5>

<item 6>
How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea (The New York Times, 2023-11-15T21:40:47+00:00)
There is the possibility of Hayes retaining a connection to the club via some sort of ambassadorial role, but it’s likely contingent upon a lack of conflict
with the USWNT role and responsibilities. Under American Todd Boehly’s co-ownership, expanding Chelsea’s profile and reach in the U.S. would make sense,
especially with USWNT internationals Catarina Macario and Mia Fishel playing their club football there — and CBS Sports holding WSL rights.

There is, for now, an immediate task for Hayes to focus on. Chelsea faces off against Real Madrid on Wednesday for their first match of the UWCL Champions
League group stage. Her full American arrival will not come for another half a year after that.

(Top photo: Julian Finney/Getty Images)
</item 6>

<item 7>
Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn (The Guardian, 2023-11-01T22:03:18+00:00)
“We should be proud of this performance.”

Blackburn certainly managed to cause problems for Chelsea even after the hosts made their intentions clear straight from the kick-off when Nicolas Jackson
forced a corner when his shot was blocked. Travis was very lucky not to concede a penalty when he slid in against Sterling after 10 minutes, with no VAR to
overrule what looked like a clear and obvious error from the referee Tim Robinson.

It took a good save from Leopold Wahlstedt to deny James but Blackburn then enjoyed a good period of possession in their opponents’ half. James was inches away
from breaking the deadlock midway through the first half but his shot just eluded the far post before Enzo Fernández forced Wahlstedt to palm away his effort
soon afterwards.

skip past newsletter promotion Sign up to Football Daily Free daily newsletter Kick off your evenings with the Guardian's take on the world of football Privacy
Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties.
</item 7>

<item 8>
Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’ (The Independent - Sports, 2023-11-15T22:32:05+00:00)
He struck in the Community Shield and the European Super Cup for City, against Arsenal and Tottenham for Chelsea. That big-match temperament may equip him for
tournament football in an England shirt. So, too, his nerveless penalty-taking.

Palmer in training at St George’s Park this week (PA)

He has, in effect, anointed himself as the spot-kick taker for the £1bn team, taking the ball at Turf Moor to open his Chelsea account. “It was just an in-game
thing, really,” he said, with a matter-of-fact stance. “Once I scored at Burnley, I just thought: ‘I’ll take them’.” If penalties have been the English disease,
Palmer has no hesitation when asked if he would be willing to come off the bench as a specialist from 12 yards for a shootout. “Yeah, I would be,” he said.
</item 8>

<item 9>
Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn (The Guardian, 2023-11-01T22:03:18+00:00)
It is no wonder Mauricio Pochettino spends most of his time watching Chelsea on the edge of the touchline with his hands clasped tightly in his pockets and a
tense expression on his face.

Even with his side 2-0 up and cruising against Blackburn thanks to goals from Benoît Badiashile and a rejuvenated Raheem Sterling, the Chelsea manager was a
bundle of nerves throughout what turned out to be a comfortable victory in the end.

But while the lack of cutting edge that has become a hallmark of the Todd Boehly-era in west London still looms large as he prepares for his first return to
table-topping Tottenham on Monday night, Pochettino could take heart from Cole Palmer’s brilliant performance as the former Manchester City forward helped to
dispatch Jon Dahl Tomasson’s inexperienced side.
</item 9>

<item 10>
Chelsea and England star Cole Palmer: ‘The most annoying thing about moving down south? The traffic’ (The Independent - Sports, 2023-11-15T22:32:05+00:00)
“I was always breaking her pipes,” he recalled. He has come a long way since then; particularly in 2023. “Yeah, it’s been crazy,” he said. “The changes have
happened so fast. From winning the Euros to signing for Chelsea then getting signed up here.”

Palmer chats with Arsenal winger Bukayo Saka (The FA/Getty)

His part in England Under-21s’ European Championship win may have caught Gareth Southgate’s eye. His immediate decision was whether he had done enough to
impress Pep Guardiola. There had been interest in taking him on loan, from Brighton and Burnley and RB Leipzig. City decided that he could only leave on a
permanent deal and Chelsea forked out £42.5m. “Who knows what would have happened if I had stayed?” Palmer mused. “Maybe I would have played more, maybe not.
But I think the decision to go to Chelsea so far is paying off.”

That is an understatement.
</item 10>
=== END ITEM id=b9e0a9c3c124ec81 ===
