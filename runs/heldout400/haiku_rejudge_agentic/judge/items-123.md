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

=== ITEM id=3fdc6a8083fec632 ===
Gold sentences and facts:
Sentence 0: The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.
  Fact 0.0: The present Board of Trustees of the Fund consists of the Secretary of Education as Chairman.
  Fact 0.1: The present Board of Trustees of the Fund includes the Director-General of the National Economic and Development Authority.
  Fact 0.2: The present Board of Trustees of the Fund includes three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP), and the Philippine Association of Colleges and Universities (PACU).
  Fact 0.3: The present Board of Trustees of the Fund shall also constitute the proposed Board of Trustees of the PhilEd.
Sentence 1: In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five directors.
  Fact 1.0: Andrew Wilkinson is 72 years old.
  Fact 1.1: Andrew Wilkinson has informed the HSF board of his intention to retire as a board director.
  Fact 1.2: Andrew Wilkinson's retirement will reduce the company's board to five directors.

Retrieved items:
<item 1>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
its share price).

For example: On September 14, HSF’s market cap stood at approximately GBP £1.125 billion, which represented a 40% discount on the firm’s operative NAV of USD
$2.316 billion as of the end of March this year.

(Said ‘operative NAV’ was calculated by an independent valuer – Citrin Cooperman – using a discount rate of 8.5%.)

This is where the importance of that potential $440 million transaction comes in.

As MBW reported earlier this month, Hipgnosis Songs Fund’s shareholders are currently considering a proposal to sell 29 of the company’s catalogs to another
Hipgnosis company – the Blackstone-backed Hipgnosis Songs Capital (HSC) – for USD $440 million.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, Hipgnosis Songs Capital, Citrin Cooperman, Blackstone, Music Business Worldwide, Andrew Sutch, continuation vote, operative NAV,
proposal to sell 29 catalogs, Andrew Sutch stepping down as Chair, Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’
likely to take place on October 26
</item 1>

<item 2>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
This has been evidenced by the 44% total return, including right to income, transaction fees and expected taxes, realised since acquisition on the [$440
million] Transaction.

“Furthermore, the Board and the Investment Adviser are committed to ensuring that this value is achieved for shareholders by a re-rating of the share price and
are determined to deliver on the ongoing opportunity of the Company.”Music Business Worldwide

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, Music Business Worldwide, Andrew Sutch, continuation vote, share price, Hipgnosis Songs Fund Chair, Andrew Sutch, to step down;
all-important ‘continuation vote’ likely to take place on October 26, Investment Adviser, Board, shareholders, Transaction
</item 2>

<item 3>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
That potential transaction is being managed for both HSF and HSC by two separate teams within their mutual investment advisor, Hipgnosis Song Management (HSM),
which is run and part-owned by Merck Mercuriadis.

When it announced its intention to explore the $440 million catalog sale to HSC two weeks ago, Hipgnosis Songs Fund confirmed that one of its motivating factors
was that the transaction could potentially act “as a catalyst for a re-rating of [HSF’s] share price”.

Aka: Increasing that share price to bring it up closer to the current operative NAV of the company.

HSC’s $440 million offer to acquire the 29 catalogs, HSF confirmed earlier this month, equates to a 26% increase on the amount HSF paid for those catalogs to
acquire them in the past.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, Music Business Worldwide, Andrew Sutch, continuation vote, operative NAV, HSC, Hipgnosis Song Management, Merck Mercuriadis,
$440 million catalog sale offer, 29 catalogs, share price, Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to
take place on October 26
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

Graph facts:
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, HSF board, valuation, Citrin Cooperman Advisors LLC, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest
valuation of the company. What’s it going to do about it?, Music Business Worldwide, Hipgnosis Song Management
</item 4>

<item 5>
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

Graph facts:
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Sale of 20,000 non-core songs from HSF’s portfolio for $23.1 million

Entities: Hipgnosis Songs Fund, Music Business Worldwide, Andrew Sutch, continuation vote, Andrew Wilkinson, AGM, Hipgnosis Songs Fund Chair, Andrew Sutch, to
step down; all-important ‘continuation vote’ likely to take place on October 26
</item 5>

<item 6>
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

Graph facts:
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Andrew Sutch -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 -MENTIONS-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund

Entities: Rob Naylor, Hipgnosis Songs Fund, Hipgnosis Songs Fund board, Citrin Cooperman, Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s
latest valuation of the company. What’s it going to do about it?, music royalties, Music Business Worldwide, JKBX
</item 6>

<item 7>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
Hipgnosis Songs Fund (HSF), the UK-listed music rights fund, looks likely to hold its all-important 2023 ‘continuation vote’ on Thursday, October 26, according
to an announcement from the firm today.

Why’s that ‘likely’ rather than ‘definitely’? Because there’s still a possibility it might get bumped into November – the reasons for which we’ll get into later
in this article.

The ‘continuation vote’ will take place at HSF’s Annual General Meeting (AGM), and will see the firm’s shareholders vote on whether to keep the entity going as
a closed-ended investment entity i.e. a publicly-traded bundle of music rights.

How will that vote go? Obviously we don’t know yet – and there’s the small matter of a $440 million transaction looming that could potentially influence things.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, Music Business Worldwide, Andrew Sutch, continuation vote, Annual General Meeting, $440 million transaction, Hipgnosis Songs
Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26
</item 7>

<item 8>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
HSF also confirmed a number of other new measures in a ‘circular’ to its shareholders today.

These measures included the fact that, if Hipgnosis Songs Fund shareholders vote to remain a publicly-traded fund (via that 2023 ‘continuation vote’) at the
firm’s next AGM, then another ‘continuation vote’ will be put in front of HSF shareholders at an Extraordinary General Meeting in January 2026.

(The ‘continuation vote’ that will take place in Q4 2023 has been scheduled ever since HSF went public in 2018 – i.e. five years after it floated. The 2026
‘Continuation Vote’, as newly-agreed, means another one is now being scheduled for around two years’ time.)

In addition, HSF today confirmed some new amendments to its ‘Investment Advisory Agreement’ (IAA) with HSM.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, HSM, Music Business Worldwide, Andrew Sutch, AGM, 2023 continuation vote, 2026 Continuation Vote, Extraordinary General Meeting,
Investment Advisory Agreement, Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26
</item 8>

<item 9>
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26 (Music Business Worldwide,
2023-09-28T23:36:54+00:00)
If a ‘superior proposal’ (versus the $440 million offer from HSC) is received during the ‘go-shop’ period, said HSF today – whether or not HSC uses its
‘matching right’ to beat it – it may be necessary to adjourn the HSF AGM (and therefore the 2023 ‘continuation vote’) into November.

In its note to shareholders today, the HSF board reiterated its belief – and Hipgnosis Song Management’s belief – that HSF owns “a unique portfolio of iconic,
culturally significant songs that will deliver strong long-term value as they benefit from the structural tailwinds in the music industry”.

The note added: “Furthermore, the [HSF] Board believes that the Investment Adviser’s approach to Song Management should enable the Company to outperform the
wider music market.

Graph facts:
- Andrew Wilkinson -WORKS_FOR-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund -SAID-> valuation
- Christopher Brown -SAID-> Hipgnosis Songs Fund
- Hipgnosis Songs Fund’s board is concerned over Citrin Cooperman’s latest valuation of the company. What’s it going to do about it? -MENTIONS-> Hipgnosis Songs
Fund
- Hipgnosis Songs Fund -PARTICIPATED_IN-> Proposed sale of assets to Hipgnosis Songs Capital for USD $440 million (pre-costs) or $417.5 million (post-costs)

Entities: Hipgnosis Songs Fund, HSF board, Music Business Worldwide, music industry, Andrew Sutch, HSC, Hipgnosis Song Management, 2023 continuation vote,
Hipgnosis Songs Fund Chair, Andrew Sutch, to step down; all-important ‘continuation vote’ likely to take place on October 26, Investment Adviser, HSF AGM, $440
million offer from HSC, Song Management, music market
</item 9>

<item 10>
Graph query row:
article: 200+ of the best deals from Amazon's Cyber Monday sale
published_on: 2023-11-27T08:45:59+00:00
publisher: Mashable
relevant_relationships: [{'entity': 'Echo', 'relationship': 'MENTIONS'}, {'entity': 'Cyber Weekend', 'relationship': 'MENTIONS'}, {'entity': 'Echo Show',
'relationship': 'MENTIONS'}, {'entity': 'smart screen', 'relationship': 'MENTIONS'}, {'entity': 'smart lights', 'relationship': 'MENTIONS'}, {'entity': 'Audible
Premium Plus', 'relationship': 'MENTIONS'}, {'entity': 'Echo smart displays', 'relationship': 'MENTIONS'}, {'entity': 'Echo smart speakers', 'relationship':
'MENTIONS'}, {'entity': 'Fire TVs', 'relationship': 'MENTIONS'}, {'entity': 'MGM+', 'relationship': 'MENTIONS'}, {'entity': 'Liquid Retina display',
'relationship': 'MENTIONS'}, {'entity': 'A14 Bionic chip', 'relationship': 'MENTIONS'}, {'entity': 'USB-C charging', 'relationship': 'MENTIONS'}, {'entity':
'Microsoft Surface Laptop Go 3', 'relationship': 'MENTIONS'}, {'entity': 'Surface Go 2', 'relationship': 'MENTIONS'}, {'entity': 'Go 3', 'relationship':
'MENTIONS'}, {'entity': 'QuietComfort 45s', 'relationship': 'MENTIONS'}, {'entity': "Amazon's Cyber Monday sale", 'relationship': 'MENTIONS'}, {'entity':
'META50', 'relationship': 'MENTIONS'}, {'entity': "Mashable's daily Deals newsletter", 'relationship': 'MENTIONS'}, {'entity': 'Terms of Use', 'relationship':
'MENTIONS'}, {'entity': 'Tablet deals', 'relationship': 'MENTIONS'}, {'entity': 'Gaming keyboards', 'relationship': 'MENTIONS'}, {'entity': 'VR headsets',
'relationship': 'MENTIONS'}, {'entity': 'Vacuum deals', 'relationship': 'MENTIONS'}, {'entity': 'Cordless vacuums', 'relationship': 'MENTIONS'}, {'entity':
'Robot vacuums and vacuum/mop hybrids', 'relationship': 'MENTIONS'}, {'entity': 'Kitchen deals', 'relationship': 'MENTIONS'}, {'entity': 'Pizza ovens',
'relationship': 'MENTIONS'}, {'entity': 'Sparkling water makers', 'relationship': 'MENTIONS'}, {'entity': 'Toaster ovens', 'relationship': 'MENTIONS'},
{'entity': 'Amazon device and service deals', 'relationship': 'MENTIONS'}, {'entity': 'Blink', 'relationship': 'MENTIONS'}, {'entity': 'Chromecast',
'relationship': 'MENTIONS'}, {'entity': 'Google Nest', 'relationship': 'MENTIONS'}, {'entity': 'Fitness deals', 'relationship': 'MENTIONS'}, {'entity':
'Smartwatches and fitness trackers', 'relationship': 'MENTIONS'}, {'entity': 'Apple deals', 'relationship': 'MENTIONS'}, {'entity': 'Beauty tech deals',
'relationship': 'MENTIONS'}, {'entity': 'Drone deals', 'relationship': 'MENTIONS'}, {'entity': 'Camera deals', 'relationship': 'MENTIONS'}, {'entity': 'GoPro',
'relationship': 'MENTIONS'}, {'entity': 'Lego deals', 'relationship': 'MENTIONS'}, {'entity': 'Gift card deals', 'relationship': 'MENTIONS'}, {'entity': 'Mac',
'relationship': 'MENTIONS'}, {'entity': 'Microsoft', 'relationship': 'MENTIONS'}, {'entity': 'Apple', 'relationship': 'MENTIONS'}, {'entity': 'AirPods',
'relationship': 'MENTIONS'}, {'entity': 'AirPods Max', 'relationship': 'MENTIONS'}, {'entity': 'Echo devices', 'relationship': 'MENTIONS'}, {'entity': 'Alexa',
'relationship': 'MENTIONS'}, {'entity': 'privacy policy', 'relationship': 'MENTIONS'}, {'entity': 'iPad', 'relationship': 'MENTIONS'}, {'entity': 'Eero',
'relationship': 'MENTIONS'}, {'entity': 'Meta Quest 2', 'relationship': 'MENTIONS'}, {'entity': 'Apple Watch', 'relationship': 'MENTIONS'}, {'entity': 'Smart
Home Deals', 'relationship': 'MENTIONS'}, {'entity': 'Speaker Deals', 'relationship': 'MENTIONS'}, {'entity': 'Bose', 'relationship': 'MENTIONS'}, {'entity':
'Earbuds', 'relationship': 'MENTIONS'}, {'entity': 'MacBooks', 'relationship': 'MENTIONS'}, {'entity': 'MacBook Air', 'relationship': 'MENTIONS'}, {'entity':
'ANC', 'relationship': 'MENTIONS'}, {'entity': 'Echo Buds', 'relationship': 'MENTIONS'}, {'entity': 'MacBook', 'relationship': 'MENTIONS'}, {'entity':
'headphones', 'relationship': 'MENTIONS'}, {'entity': 'Fire TV', 'relationship': 'MENTIONS'}, {'entity': 'Black Friday', 'relationship': 'MENTIONS'}, {'entity':
'Kindle', 'relationship': 'MENTIONS'}, {'entity': 'Prime', 'relationship': 'MENTIONS'}, {'entity': 'Gaming mice', 'relationship': 'MENTIONS'}, {'entity': 'Black
Friday sale', 'relationship': 'MENTIONS'}, {'entity': 'Intel', 'relationship': 'MENTIONS'}, {'entity': 'Amazon', 'relationship': 'MENTIONS'}, {'entity': 'Cyber
Monday sale', 'relationship': 'MENTIONS'}, {'entity': 'Intel Core i5 processor', 'relationship': 'MENTIONS'}, {'entity': 'Fire tablets', 'relationship':
'MENTIONS'}, {'entity': 'gaming deals', 'relationship': 'MENTIONS'}, {'entity': 'Amazon credit', 'relationship': 'MENTIONS'}, {'entity': 'gaming headsets',
'relationship': 'MENTIONS'}, {'entity': 'Apple Watches', 'relationship': 'MENTIONS'}]
page: None
passage: None
</item 10>
=== END ITEM id=3fdc6a8083fec632 ===

=== ITEM id=81a346cf3ca6f28a ===
Gold sentences and facts:
Sentence 0: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 0.0: Google gets other benefits from Android.
  Fact 0.1: Google could solve this problem today if it wanted to.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will suggest that Google hid items from discovery by allowing execs and employees to have 'off the record' chats.
  Fact 1.1: Google will counter that suggestion.
  Fact 1.2: Google will say it has provided the court with thousands of chat logs.
  Fact 1.3: Google will say it has provided the court with millions of documents.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Alphabet
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -MENTIONS-> Alphabet
- ASX set to rise ahead of inflation report after big tech lifts Wall Street -MENTIONS-> Alphabet
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> Alphabet
- Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas -MENTIONS-> Alphabet

Entities: Alphabet, Helena World Chronicle, U.S. District Court in D.C., Search Generative Experience, Bard AI chatbot, Knowledge Graph, anticompetitive
behavior, U.S. antitrust law, Sherman Act, Arkansas, question-and-answer technologies, News publisher files class action antitrust suit against Google, citing
AI’s harms to their bottom line, class action lawsuit, TechCrunch, AI technologies
</item 1>

<item 2>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- 5 things we learned so far about the Google antitrust case -MENTIONS-> FTC-Facebook case
- Amit Mehta -PARTICIPATED_IN-> US v Google trial

Entities: Amazon antitrust lawsuit, FTC-Facebook case, Google antitrust lawsuit with Match Group, Google trial with Epic Games, Anticompetitive and unfair
strategies to maintain monopoly power, Vertical search, Pre-installs, Third-party apps, Commission structure, More from the US v Google trial: vertical search,
pre-installs and the case of Firefox/Yahoo, US v Google trial, Yahoo, Firefox, anticompetitive behavior, Google Play
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
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

Graph facts:
- Alphabet -ACCUSED_OF-> Sherman Act
- Alphabet -ACCUSED_OF-> U.S. antitrust law
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. antitrust law
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Sherman Act
- Epic v. Google, explained -MENTIONS-> User Choice Billing

Entities: U.S. antitrust law, Sherman Act, User Choice Billing, Google first-party billing system, Epic Games’ antitrust battle with Google, California’s
anticompetitive laws, Cartwright Act, Unfair Competition Law, distribution of apps to Android users, payment processing solutions for content inside Android
apps, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, TechCrunch, Google
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.
- Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
-MENTIONS-> Washington Post
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Search Generative Experience

Entities: SGE, Google Bard AI, GoogleBot, Google general search service, AdSense, News Media Alliance, Washington Post, Epic Games lawsuit against Google over
app store antitrust issues, app store antitrust issues, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line,
class action antitrust suit against Google, Epic Games, Allen Institute for AI, TechCrunch, Google
</item 5>

<item 6>
Graph query row:
article: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
publisher: TechCrunch
relationship: MENTIONS
connected: Alphabet
</item 6>

<item 7>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Chrome, Expedia, Yahoo, Firefox,
Android OEMs, antitrust, Microsoft, TechCrunch, Google, Justice Department, iPhones, U.S.
</item 7>

<item 8>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Bing
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Bing
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> monopoly power
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Bing

Entities: monopoly power, FTC-Facebook case, Bing, Apple, WhatsApp, TechCrunch, data, Google antitrust case, 5 things we learned so far about the Google
antitrust case, DOJ, FTC, FTC lawsuit against Amazon, anticompetitive and unfair strategies, DOJ investigation of Apple, Apple's policy for third-party apps on
its devices
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: Vertical search, More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, LifeLock, Jamie
Rosenberg, Equifax data breach, iPhone 8, iPhone 8 official release, identity theft protection, competition between Google and Apple, Yahoo, Firefox, Google
Play, Android, Apple
</item 9>

<item 10>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amazon antitrust lawsuit
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Amit Mehta
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> APIs
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> FTC-Facebook case

Entities: Vertical search, More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo, US v Google trial, Rosenberg, mobile
app distribution agreement, Oppo, Chrome, Play, revenue share agreements, best in class, Yahoo, Firefox, Android, Search, Apple
</item 10>
=== END ITEM id=81a346cf3ca6f28a ===

=== ITEM id=8e52046652de8fe0 ===
Gold sentences and facts:
Sentence 0: One of the major sources of Gen Z’s distress, of course, is climate change.
  Fact 0.0: Climate change is one of the major sources of Gen Z's distress.
Sentence 1: In other words, first, we need to improve our understanding of drivers of sea surface temperature patterns—decadal coupled variability; the pace and spatial structure of ocean heat uptake; and forcing by aerosols, greenhouse gases, and volcanoes—as well as the relative timing of these drivers.
  Fact 1.0: We need to improve our understanding of the drivers of sea surface temperature patterns.
  Fact 1.1: Improving the understanding of drivers of sea surface temperature patterns is the first step.
  Fact 1.2: One driver of sea surface temperature patterns is decadal coupled variability.
  Fact 1.3: Another driver is the pace of ocean heat uptake.
  Fact 1.4: Another driver is the spatial structure of ocean heat uptake.
  Fact 1.5: Another driver is forcing by aerosols.
  Fact 1.6: Another driver is forcing by greenhouse gases.
  Fact 1.7: Another driver is forcing by volcanoes.
  Fact 1.8: We need to improve our understanding of the relative timing of these drivers.

Retrieved items:
<item 1>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Winton, M., K. Takahashi, and I. M. Held (2010), Importance of ocean heat uptake efficacy to transient climate change, J. Clim., 23, 2,333–2,344,
https://doi.org/10.1175/2009JCLI3139.1.

Wills, R. C. J., et al. (2022), Systematic climate model biases in the large-scale patterns of recent sea-surface temperature and sea-level pressure change,
Geophys. Res. Lett., 49, e2022GL100011, https://doi.org/10.1029/2022GL100011.

Zhou, C., et al. (2021), Greater committed warming after accounting for the pattern effect, Nat. Clim. Change, 11, 132–136,
https://doi.org/10.1038/s41558-020-00955-x.

Graph facts:
- La desigualdad del estrés por calor -PUBLISHED_BY-> Eos: Earth And Space Science News
- Patterns of Surface Warming Matter for Climate Sensitivity -PUBLISHED_BY-> Eos
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> South America
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> climate science
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> aerosol-cloud interactions

Entities: Eos: Earth And Space Science News, Patterns of Surface Warming Matter for Climate Sensitivity, Geophys. Res. Lett., Importance of ocean heat uptake
efficacy to transient climate change, J. Clim., Winton, M., K. Takahashi, I. M. Held, Systematic climate model biases in the large-scale patterns of recent
sea-surface temperature and sea-level pressure change, Wills, R. C. J., Greater committed warming after accounting for the pattern effect, Nat. Clim. Change,
Zhou, C.
</item 1>

<item 2>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Feedbacks calculated from observations or from atmosphere-only model simulations forced with the observed surface warming pattern over the past couple of
decades imply less warming than those from model simulations with a fully interactive ocean, which have the freedom to create their own surface warming patterns
[e.g., Gregory et al., 2020].

The third strand of research came from oceanography, showing that the atmospheric cooling effect of ocean heat uptake differs depending on where it occurs: One
unit of ocean heat uptake in high latitudes cools Earth more effectively than the same unit taken up by the low-latitude oceans. This difference is relevant
because the largest heat uptake by the ocean occurs at higher latitudes. The effect, termed ocean heat uptake efficacy, turns out to be another manifestation of
the dependence of radiative feedbacks on surface temperature patterns [Winton et al., 2010; Lin et al., 2021].
</item 2>

<item 3>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
And most pressing for the communities relying on the coupled models’ climate change projections, To what extent are the remaining carbon budgets [Zhou et al.,
2021] with respect to global warming targets, rates of near-term future warming, and detection and attribution efforts contingent on the pattern effect?

A Collaborative Outlook

Working to solve the problems outlined above at the intersection of different disciplines and with input from different research communities will likely benefit
related research into, for example, the sensitivity of tropical cyclones and future rainfall changes in the southwestern United States or in South America to
surface warming patterns.

Promising new tools and observations are beginning to emerge. They include targeted model experiments and intercomparisons, extended global satellite
observations of clouds and radiation, Argo floats sampling deep-ocean heat uptake, estimates of radiative imbalances before the year 2000, and new observational
constraints on clouds’ sensitivity to environmental controlling factors. These developments encourage an optimistic outlook on our ability to quantify the
pattern effect and its implications over the next couple of years.

Graph facts:
- The shipping industry’s boom-and-bust cycle is so severe carriers face going from bumper profits to losing money in the space of a few years: 'I'm certainly
concerned' -MENTIONS-> South America
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> South America
- Day of Reckoning -- featuring Anthony Joshua, Deontay Wilder: Predictions, undercard, odds, picks, start time -MENTIONS-> South America
- La desigualdad del estrés por calor -PUBLISHED_BY-> Eos: Earth And Space Science News
- The best comedies you’ll find on streaming right now -MENTIONS-> South America

Entities: Eos: Earth And Space Science News, South America, pattern effect, Surface warming, surface warming patterns, Patterns of Surface Warming Matter for
Climate Sensitivity, coupled models’ climate change projections, remaining carbon budgets, global warming targets, rates of near-term future warming, detection
and attribution efforts, tropical cyclones, future rainfall changes, southwestern United States, targeted model experiments and intercomparisons
</item 3>

<item 4>
Graph query row:
article: Patterns of Surface Warming Matter for Climate Sensitivity
publisher: Eos
article_says_about: []
</item 4>

<item 5>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The paleorecord could further be crucial for understanding the timescales and relative importance of internal variability, forced response, and extratropical
forcing to rates of warming or cooling in the equatorial Pacific in nature and in climate models.

Should We Trust Model Patterns in Climate Change Scenarios?

Our understanding of the pattern effect raises the question of whether climate models can reproduce observed warming patterns. Coupled climate models simulate a
freely evolving ocean surface and hence have their own expression of internal variability. For example, we would not expect a coupled model to simulate El Niño
events at the same time as they happened in nature, except by chance. This feature complicates comparisons between models and observations, especially on
decadal timescales and beyond, for which we know little about the spectrum of internal variability in the real world.

Unfortunately, coupled climate models seem unable to simulate observed surface warming patterns across some key regions in which surface sensitivity modulates
clouds, even when they account for internal variability.

Graph facts:
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> South America
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> climate science
- La desigualdad del estrés por calor -PUBLISHED_BY-> Eos: Earth And Space Science News
- Patterns of Surface Warming Matter for Climate Sensitivity -PUBLISHED_BY-> Eos
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> aerosol-cloud interactions

Entities: Eos: Earth And Space Science News, clouds, climate models, pattern effect, paleorecord, internal variability, forced response, extratropical forcing,
equatorial Pacific, El Niño, observed warming patterns, climate change scenarios, ocean surface, surface sensitivity, surface warming patterns
</item 5>

<item 6>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
In other words, first, we need to improve our understanding of drivers of sea surface temperature patterns—decadal coupled variability; the pace and spatial
structure of ocean heat uptake; and forcing by aerosols, greenhouse gases, and volcanoes—as well as the relative timing of these drivers. An important goal is
to explain historical patterns quantitatively and to predict their likely future evolution. Second, we need to quantify the dependence of local and remote
top-of-the-atmosphere radiative fluxes on the magnitude, spatial scale, and sign (positive or negative) of surface temperature changes. Third, we have to
explain how ocean heat uptake and radiative feedback are connected on various timescales, both globally and regionally.

Interpreting Paleorecords Requires Caution

In the same way that the historical evolution of surface warming and the planet’s radiation budget inform Earth’s future only to a limited extent, analyses of
paleorecords for recent and deep time intervals must account for the pattern effect to be applicable to climate projections of the future.

Graph facts:
- Patterns of Surface Warming Matter for Climate Sensitivity -PUBLISHED_BY-> Eos
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> South America
- La desigualdad del estrés por calor -PUBLISHED_BY-> Eos: Earth And Space Science News
- The dominant contribution of Southern Ocean heat uptake to time-evolving radiative feedback in CESM -MENTIONS-> radiative feedback
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> climate science

Entities: Eos: Earth And Space Science News, pattern effect, ocean heat uptake, radiative feedback, sea surface temperature patterns, decadal coupled
variability, aerosols, volcanoes, top-of-the-atmosphere radiative fluxes, surface temperature changes, paleorecords, climate projections, radiation budget,
Patterns of Surface Warming Matter for Climate Sensitivity, climate sensitivity
</item 6>

<item 7>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
But they’re not just reading about the detrimental effects of human-caused global warming, they’re living through the consequences themselves.

This summer reached record-breaking temperatures, with July being the hottest month the planet has seen in over 100,000 years. As a result, Arizona experienced
a monthlong heat wave with temperatures at or above 110 degrees every day. Deadly fires broke out across the Mediterranean; suffocating smoke from Canadian
wildfires blanketed New York City and the Northeast for days; and ice melt in the Arctic accelerated.

And extreme heat is likely here to stay—and get worse—unless countries can rapidly reduce their carbon emissions. That’s why Gen Z is more concerned with
sustainability than any generation before them.

Graph facts:
- Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? -MENTIONS-> New York City
- The people who ruined the internet -MENTIONS-> New York City
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> New York City
- Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance -MENTIONS-> New York City
- The best video games of the year so far -MENTIONS-> New York City

Entities: New York City, Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images', war in
Israel, disturbing images, human-caused global warming, monthlong heat wave, fires across the Mediterranean, Canadian wildfires, Northeast, Arctic, ice melt,
Fortune, Arizona, carbon emissions, sustainability
</item 7>

<item 8>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Acknowledgments

The authors thank all participants of the US CLIVAR pattern effect workshop. M.Z.’s work was supported by the U.S. Department of Energy (DOE) Regional and
Global Model Analysis program area and was performed under the auspices of DOE by Lawrence Livermore National Laboratory under contract DE-AC52-07NA27344. P.C.
was supported by the UK Natural Environment Research Council, grants NE/T006250/1 and NE/V012045/1. T.A. was supported by the Met Office Hadley Centre Climate
Programme funded by the Department of Business, Energy and Industrial Strategy and received funding from the European Union’s Horizon 2020 research and
innovation program under grant agreement 820829. M.R. was supported by NASA under grant 80NSSC21K1042.

References

Andrews, T., et al. (2022), On the effect of historical SST patterns on radiative feedback, J. Geophys. Res.

Graph facts:
- La desigualdad del estrés por calor -PUBLISHED_BY-> Eos: Earth And Space Science News
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> South America
- The dominant contribution of Southern Ocean heat uptake to time-evolving radiative feedback in CESM -MENTIONS-> radiative feedback
- Patterns of Surface Warming Matter for Climate Sensitivity -PUBLISHED_BY-> Eos
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> climate science

Entities: Eos: Earth And Space Science News, radiative feedback, Patterns of Surface Warming Matter for Climate Sensitivity, M.Z., P.C., T.A., M.R., US CLIVAR
pattern effect workshop, U.S. Department of Energy, Regional and Global Model Analysis program area, UK Natural Environment Research Council, Met Office Hadley
Centre Climate Programme, Department of Business, Energy and Industrial Strategy, Horizon 2020 research and innovation program, On the effect of historical SST
patterns on radiative feedback
</item 8>

<item 9>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
“No solo necesitamos entender mejor dónde hace calor, sino también cuáles son los factores socioeconómicos para los habitantes de esas zonas”, dijo Whitely
Binder. “Podemos comenzar a analizar lo que hay en las islas de calor y utilizarlo para comenzar a informar las decisiones políticas que vamos a tomar”.

—Rebecca Owen (@beccapox), Escritora de ciencia

This translation by Nel Rodriguez Sepulveda was made possible by a partnership with Planeteando. Esta traducción fue posible gracias a una asociación con
Planeteando.

This news article is included in our ENGAGE resource for educators seeking science news for their classroom lessons. Browse all ENGAGE articles , and share with
your fellow educators how you integrated the article into an activity in the comments section below.

Text © 2023. The authors. CC BY-NC-ND 3.0

Except where otherwise noted, images are subject to copyright.

Graph facts:
- La desigualdad del estrés por calor -PUBLISHED_BY-> Eos
- La desigualdad del estrés por calor -MENTIONS-> Washington
- La desigualdad del estrés por calor -MENTIONS-> Brooklyn
- La desigualdad del estrés por calor -MENTIONS-> Eos
- La desigualdad del estrés por calor -MENTIONS-> One Earth

Entities: La desigualdad del estrés por calor, Eos: Earth And Space Science News, desigualdad del estrés por calor, Whitely Binder, Rebecca Owen, Nel Rodriguez
Sepulveda, Planeteando, ENGAGE resource, factores socioeconómicos, islas de calor, decisiones políticas
</item 9>

<item 10>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Author Information

Maria Rugenstein (maria.rugenstein@colostate.edu), Colorado State University, Fort Collins; Mark Zelinka, Lawrence Livermore National Laboratory, Livermore,
Calif.; Kristopher B. Karnauskas, University of Colorado Boulder; Paulo Ceppi, Imperial College London, London; and Timothy Andrews, Met Office Hadley Centre,
Exeter, U.K.

Citation: Rugenstein, M., M. Zelinka, K. B. Karnauskas, P. Ceppi, and T. Andrews (2023), Patterns of surface warming matter for climate sensitivity, Eos, 104,
https://doi.org/10.1029/2023EO230411. Published on 31 October 2023.

Text © 2023. The authors. CC BY-NC-ND 3.0

Except where otherwise noted, images are subject to copyright. Any reuse without express permission from the copyright owner is prohibited.

Graph facts:
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> South America
- La desigualdad del estrés por calor -MENTIONS-> Eos
- La desigualdad del estrés por calor -PUBLISHED_BY-> Eos
- Patterns of Surface Warming Matter for Climate Sensitivity -PUBLISHED_BY-> Eos: Earth And Space Science News
- Patterns of Surface Warming Matter for Climate Sensitivity -MENTIONS-> climate science

Entities: Eos, Surface warming, Patterns of Surface Warming Matter for Climate Sensitivity, climate sensitivity, Maria Rugenstein, Mark Zelinka, Kristopher B.
Karnauskas, Paulo Ceppi, Timothy Andrews, Colorado State University, Lawrence Livermore National Laboratory, University of Colorado Boulder, Imperial College
London, Met Office Hadley Centre, Fort Collins
</item 10>
=== END ITEM id=8e52046652de8fe0 ===
