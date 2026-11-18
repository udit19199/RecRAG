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

This file has 2 items.

=== ITEM id=64ee3813e99a9f83 ===
Gold sentences and facts:
Sentence 0: The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.
  Fact 0.0: The present Board of Trustees of the Fund consists of the Secretary of Education as Chairman.
  Fact 0.1: The present Board of Trustees of the Fund includes the Director-General of the National Economic and Development Authority.
  Fact 0.2: The present Board of Trustees of the Fund includes the President of the Association of Christian Schools, Colleges and Universities (ACSCU).
  Fact 0.3: The present Board of Trustees of the Fund includes the President of the Catholic Educational Association of the Philippines (CEAP).
  Fact 0.4: The present Board of Trustees of the Fund includes the President of the Philippine Association of Colleges and Universities (PACU).
  Fact 0.5: The present Board of Trustees of the Fund shall also constitute the proposed Board of Trustees of the PhilEd.
Sentence 1: In addition, Andrew Wilkinson, 72, has informed the HSF board of his intention to retire as a board director, which will reduce the company’s board to five directors.
  Fact 1.0: Andrew Wilkinson is 72 years old.
  Fact 1.1: Andrew Wilkinson has informed the HSF board of his intention to retire as a board director.
  Fact 1.2: Andrew Wilkinson's retirement will reduce the HSF board to five directors.

Retrieved items:
<item 1>
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
</item 3>

<item 4>
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
</item 6>

<item 7>
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
From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
The present Board of Trustees of the Fund which consists of the Secretary of Education as Chairman with the Director-General of the National Economic and
Development Authority, the three Presidents of the Association of Christian Schools, Colleges and Universities (ACSCU), the Catholic Educational Association of
the Philippines (CEAP) and the Philippine Association of Colleges and Universities (PACU) shall also constitute the proposed Board of Trustees of the PhilEd.

The management team of PEAC will be the incoming management team of PhilEd.

Interestingly, PhilHealth, in addition to premiums collected from members as well as subsidies from local governments who have enrolled indigents in the
program, receives a share of the “sin taxes” collected from Tobacco and Sugar Sweetened Beverage. The justification for this is that tobacco smokers and sugar
sweetened beverage drinkers impose an additional burden on the healthcare system of the Filipinos and so must contribute to their health services. In Fiscal
Year 2022, this amounted to P74 billion.
</item 9>

<item 10>
Graph query row:
article: From FAPE/PEAC to PhilEd
passage: From FAPE/PEAC to PhilEd (Business World, 2023-11-12T16:04:21+00:00)
We recently presented a paper at the 5th Paderanga-Varela Memorial Lecture entitled “Alternate Service Delivery Systems: Health and Education.” In that lecture,
we looked at the Department of Health (DoH) and the Department of Education (DepEd) as service delivery systems, delivering healthcare and education
respectively.

We concluded that the service delivery system of the DoH is far superior to the DepEd for two main reasons. The first is that the DoH has devolved its hospitals
to the local government units while the DepEd has not. Secondly, the DoH has spun off its health financing into the Philippine Health Insurance Corp.
(PhilHealth) while the DepEd has not. In this article we argue for the creation of the Philippine Education Development Fund (PhilEd).

When Isidro Consunji, President of DMCI Holdings, Inc. was asked by his officers what the role of his Chief Financial Officer (CFO) would be, he gave a
three-word answer, “Fund my dreams.”
</item 10>
=== END ITEM id=64ee3813e99a9f83 ===

=== ITEM id=ff0f73b0ab68033e ===
Gold sentences and facts:
Sentence 0: For all the other benefits they get from Android, Google could solve this problem today if they wanted to.
  Fact 0.0: They get other benefits from Android.
  Fact 0.1: Google could solve this problem today if Google wanted to.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will also suggest that Google hid items from discovery.
  Fact 1.1: Epic will suggest Google hid items from discovery by allowing execs and employees to have 'off the record' chats.
  Fact 1.2: Google will counter this as well.
  Fact 1.3: Google will say it has provided the court with thousands of chat logs.
  Fact 1.4: Google will say it has provided the court with millions of documents.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Sundar Pichai
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sundar Pichai
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Sundar Pichai
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Sundar Pichai
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Sundar Pichai

Entities: Sundar Pichai, Department of Justice, antitrust, Apple, online search, TechCrunch, New York, California, U.S. government, digital advertising, U.S.
government’s antitrust trial against Google, 2020 antitrust claim, January filing, eight states, 5 things we learned so far about the Google antitrust case
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
published_on: 2023-12-15T17:56:02+00:00
connected_entities: [{'entity': 'TechCrunch', 'labels': ['__KGBuilder__', '__Entity__', 'Publisher'], 'relationship': 'PUBLISHED_BY'}, {'entity': 'Alphabet',
'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Helena World Chronicle', 'labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'U.S. District Court in D.C.', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'entity': 'Search Generative Experience', 'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'},
{'entity': 'Bard AI chatbot', 'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity': 'Knowledge Graph', 'labels':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity': 'anticompetitive behavior', 'labels': ['__KGBuilder__', '__Entity__',
'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'U.S. antitrust law', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'},
{'entity': 'Sherman Act', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'Arkansas', 'labels': ['__KGBuilder__',
'__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': 'question-and-answer technologies', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'],
'relationship': 'MENTIONS'}, {'entity': 'Publishers', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity':
'Knowledge Panel', 'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity': 'Featured Snippets', 'labels':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity': 'class action antitrust suit', 'labels': ['__KGBuilder__', '__Entity__',
'Event'], 'relationship': 'MENTIONS'}, {'entity': 'misappropriation of Publishers’ content', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship':
'MENTIONS'}, {'entity': 'The Wall Street Journal', 'labels': ['__KGBuilder__', '__Entity__', 'Publisher'], 'relationship': 'MENTIONS'}, {'entity': 'report on
Thursday', 'labels': ['__KGBuilder__', 'Article', '__Entity__'], 'relationship': 'MENTIONS'}, {'entity': 'deal to license news for AI model training', 'labels':
['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': 'SGE', 'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship':
'MENTIONS'}, {'entity': 'Google Bard AI', 'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity': 'GoogleBot', 'labels':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity': 'Google general search service', 'labels': ['__KGBuilder__', '__Entity__',
'Product'], 'relationship': 'MENTIONS'}, {'entity': 'AdSense', 'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'entity':
'News Media Alliance', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Washington Post', 'labels':
['__KGBuilder__', '__Entity__', 'Publisher'], 'relationship': 'MENTIONS'}, {'entity': 'Epic Games lawsuit against Google over app store antitrust issues',
'labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': 'app store antitrust issues', 'labels': ['__KGBuilder__',
'__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'U.S. Justice Department', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'entity': 'Hausfeld', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Helena
World Chronicle, LLC', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'class action antitrust suit against
Google', 'labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': 'U.S. Justice Department lawsuit against Google', 'labels':
['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': '2020 Justice Department civil antitrust suit over search and search
advertising', 'labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': 'digital ad technologies', 'labels': ['__KGBuilder__',
'__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'Clayton Act', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'},
{'entity': 'search and search advertising', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'digital news and
reference publishing', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'class-wide monetary and injunctive relief',
'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'free marketplace of ideas', 'labels': ['__KGBuilder__',
'__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'free press', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'},
{'entity': 'Epic Games', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'antitrust', 'labels':
['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'The Atlantic', 'labels': ['__KGBuilder__', '__Entity__', 'Publisher'],
'relationship': 'MENTIONS'}, {'entity': 'Axel Springer', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity':
'Allen Institute for AI', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'class action lawsuit', 'labels':
['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'entity': 'consumers', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'],
'relationship': 'MENTIONS'}, {'entity': 'AI', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'AI technologies',
'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'Similarweb', 'labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'entity': 'Google', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'entity': 'artificial intelligence', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'entity': 'D.C.', 'labels':
['__KGBuilder__', '__Entity__', 'Place'], 'relationship': 'MENTIONS'}, {'entity': 'OpenAI', 'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'entity': 'labor', 'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}]
evidence_passages: ['News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch,
2023-12-15T17:56:02+00:00)\nA new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of
anticompetitive behavior in violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based
publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also
specifically cites new AI technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.\n\nIn the complaint,
Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing publishers’
content on Google, losing them “billions of dollars.”\n\nIn addition to new AI technologies, the suit points to Google’s older question-and-answer technologies,
like the “Knowledge Graph” launched in May 2012, as part of the problem.', "News publisher files class action antitrust suit against Google, citing AI’s harms
to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\n“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right
of the search results. This panel contains a summary of content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive
database by extracting information from Publishers’ websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed
databases,'” it says.\n\nBy 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that
Google tapped into was content “misappropriated from Publishers,” the complaint alleges.\n\nOther Google technologies, like “Featured Snippets” where Google
algorithmically extracts answers from webpages, were also cited as shifting traffic away from publishers’ websites.\n\nMore importantly, perhaps, is the suit’s
tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The Wall Street Journal, which led with a
shocking statistic.", 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch,
2023-12-15T17:56:02+00:00)\nWhen online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the
AI would answer the user’s query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic
going forward, as Google today drives nearly 40% of their traffic, according to data from Similarweb.\n\nSome publishers are now trying to get ahead of the
problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model training. But overall, publishers believe they’ll
lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s report noted.\n\nThe lawsuit reiterates this concern,
claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from visiting the websites of Class members
who are part of the digital news and publishing line of commerce.”', 'News publisher files class action antitrust suit against Google, citing AI’s harms to
their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\nSGE, it argues, offers web searchers a way to seek information in a conversational mode, but
ultimately keeps users in Google’s “walled garden” as it “plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as
Google’s general search service, GoogleBot.\n\nPlus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,”
citing both a 2023 report from the News Media Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with
researchers at the Allen Institute for AI, had found that News and Media sites were the third largest category of AI training data.)\n\nThe case points to other
concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat messages — an issue
raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.', 'News publisher files class action antitrust suit
against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\nThe case also arrives alongside the filing of the U.S. Justice
Department’s lawsuit against Google for monopolizing digital ad technologies, and references the 2020 Justice Department’s civil antitrust suit over search and
search advertising (which are different markets from digital ad technologies in the more recent suit).\n\n“The anticompetitive effects of Google’s scheme cause
profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an announcement posted to the website of the law firm handling the
case, Hausfeld.\n\n“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore
and ensure competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial
intelligence,” it states.\n\nA Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.']
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 8>

<item 9>
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
=== END ITEM id=ff0f73b0ab68033e ===
