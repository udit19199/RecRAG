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

=== ITEM id=07dad088ed1419d5 ===
Gold sentences and facts:
Sentence 0: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 0.0: The filing will attempt to argue that Google restrains competition within two separate markets.
  Fact 0.1: One of the two markets is the distribution of apps to Android users.
  Fact 0.2: The other of the two markets is the market for payment processing solutions for content inside Android apps.
Sentence 1: Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence.
  Fact 1.0: Google's public documentation says Google uses three categories to rank results that are local to a searcher.
  Fact 1.1: The three categories are relevance, distance, and prominence.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that this siphoning happens through anticompetitive means.

Retrieved items:
<item 1>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 1>

<item 2>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
“Android’s choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users
safe from harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by
Wilson White, VP, Government Affairs & Public Policy.

Epic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with Match last
week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will include
testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other Android
developers.

Google is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Sundar Pichai
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Sundar Pichai
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Sundar Pichai
- 5 things we learned so far about the Google antitrust case -MENTIONS-> Sundar Pichai
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Sundar Pichai

Entities: Sundar Pichai, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, Wilson White, Match,
U.S. attorneys general, Epic's antitrust lawsuit, U.S. District Court in California’s Northern District, Department of Justice, Google's antitrust lawsuit with
the Department of Justice, search monopoly, antitrust, Netflix, Apple
</item 2>

<item 3>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Android

Entities: Android, Galaxy Store, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, OnePlus, Epic Games app, Google
Play Store, Android app developers, Project Hug, Google Games Velocity Program, antitrust battle, agreement with OnePlus, deal between Google and Samsung,
Google developer agreements, anti-competitive practices, Epic Games
</item 3>

<item 4>
Graph query row:
article: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
published_on: 2023-12-15T17:56:02+00:00
publisher: TechCrunch
passages: ['News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\nA
new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in violation
of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle, argues
that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI technologies
like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.\n\nIn the complaint, Helena World Chronicle, which owns and
publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing publishers’ content on Google, losing them “billions of
dollars.”\n\nIn addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May
2012, as part of the problem.', "News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch,
2023-12-15T17:56:02+00:00)\n“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel
contains a summary of content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information
from Publishers’ websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.\n\nBy 2020, the
Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was content
“misappropriated from Publishers,” the complaint alleges.\n\nOther Google technologies, like “Featured Snippets” where Google algorithmically extracts answers
from webpages, were also cited as shifting traffic away from publishers’ websites.\n\nMore importantly, perhaps, is the suit’s tackling of how AI will impact
publishers’ businesses. The problem was recently detailed in a report on Thursday by The Wall Street Journal, which led with a shocking statistic.", 'News
publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\nWhen online magazine
The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s query without requiring
a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today drives nearly 40% of
their traffic, according to data from Similarweb.\n\nSome publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked
a deal with OpenAI to license its news for AI model training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic
when Google’s AI products fully roll out, The WSJ’s report noted.\n\nThe lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based
search were implemented with “the goal of discouraging end-users from visiting the websites of Class members who are part of the digital news and publishing
line of commerce.”']
mentioned_entities: ['Alphabet', 'Helena World Chronicle', 'U.S. District Court in D.C.', 'Search Generative Experience', 'Bard AI chatbot', 'Knowledge Graph',
'anticompetitive behavior', 'U.S. antitrust law', 'Sherman Act', 'Arkansas', 'question-and-answer technologies', 'Publishers', 'Knowledge Panel', 'Featured
Snippets', 'class action antitrust suit', 'misappropriation of Publishers’ content', 'The Wall Street Journal', 'report on Thursday', 'deal to license news for
AI model training', 'SGE', 'Google Bard AI', 'GoogleBot', 'Google general search service', 'AdSense', 'News Media Alliance', 'Washington Post', 'Epic Games
lawsuit against Google over app store antitrust issues', 'app store antitrust issues', 'U.S. Justice Department', 'Hausfeld', 'Helena World Chronicle, LLC',
'class action antitrust suit against Google', 'U.S. Justice Department lawsuit against Google', '2020 Justice Department civil antitrust suit over search and
search advertising', 'digital ad technologies', 'Clayton Act', 'search and search advertising', 'digital news and reference publishing', 'class-wide monetary
and injunctive relief', 'free marketplace of ideas', 'free press', 'Epic Games', 'antitrust', 'The Atlantic', 'Axel Springer', 'Allen Institute for AI', 'class
action lawsuit', 'consumers', 'AI', 'AI technologies', 'Similarweb', 'Google', 'artificial intelligence', 'D.C.', 'OpenAI', 'labor']
connections: [{'from': 'Google', 'to': 'Gary Wang', 'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'Gmail', 'relationship': 'MADE'}, {'from': 'Google',
'to': 'Gmail', 'relationship': 'OWNS'}, {'from': 'Google', 'to': 'Pixel 8', 'relationship': 'MADE'}, {'from': 'AI', 'to': 'Sam Altman backs teens’ startup,
Google unveils the Pixel 8 and TikTok tests an ad-free tier', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Sam Altman backs teens’ startup, Google
unveils the Pixel 8 and TikTok tests an ad-free tier', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Cue', 'relationship': 'SAID'}, {'from': 'Google',
'to': 'Google-Apple revenue-share deal', 'relationship': 'PARTICIPATED_IN'}, {'from': 'antitrust', 'to': 'Apple defends Google Search deal in court: ‘There
wasn’t a valid alternative’', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Apple defends Google Search deal in court: ‘There wasn’t a valid
alternative’', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'John Schmidtlein', 'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'Google
antitrust lawsuit with Match Group', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Epic Games', 'to': 'Google trial with Epic Games', 'relationship':
'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Google trial with Epic Games', 'relationship': 'PARTICIPATED_IN'}, {'from': 'anticompetitive behavior', 'to':
'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'relationship': 'MENTIONS'}, {'from': 'Epic Games', 'to': 'More
from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'relationship': 'MENTIONS'}, {'from': 'antitrust', 'to': 'More from
the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'More from the US v
Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'US v Google trial',
'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'US v Google trial', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'Jamie Rosenberg',
'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'Rosenberg', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Chrome', 'relationship': 'MADE'},
{'from': 'Google', 'to': 'Play', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'Barry Diller', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Sundar
Pichai', 'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'Google travel listings', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'Mitchell Baker',
'relationship': 'SAID'}, {'from': 'Google', 'to': 'Yahoo', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to': 'Why Mozilla is betting on a decentralized
social networking future', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Hinton', 'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'Alan
Eustace', 'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'auction', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'How ego and fear
fuelled the rise of artificial intelligence', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Larry Page', 'relationship': 'FOUNDED'}, {'from': 'Google',
'to': 'Sergey Brin', 'relationship': 'FOUNDED'}, {'from': 'Google', 'to': 'Suleyman', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Google Search',
'relationship': 'MADE'}, {'from': 'Google', 'to': 'Google Search', 'relationship': 'OWNS'}, {'from': 'Google', 'to': 'Kent Walker', 'relationship':
'WORKS_FOR'}, {'from': 'Google', 'to': 'Wired opinion article', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Is Google Search better than the rest?
And is that fair?', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Is Google Search better than the rest? And is that fair?', 'relationship': 'SAID'},
{'from': 'Google', 'to': 'DuckDuckGo', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to': 'quality tests', 'relationship': 'PARTICIPATED_IN'}, {'from':
'Google', 'to': 'Megan Grey', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Alphabet', 'relationship': 'OWNS'}, {'from': 'Google', 'to': 'Alphabet',
'relationship': 'PART_OF'}, {'from': 'Google', 'to': 'Helena World Chronicle', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'anticompetitive behavior',
'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'U.S. antitrust law', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'Sherman Act',
'relationship': 'ACCUSED_OF'}, {'from': 'class action antitrust suit', 'to': 'Publishers', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Publishers', 'to':
'class action antitrust suit', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'class action antitrust suit', 'relationship': 'ACCUSED_OF'},
{'from': 'Google', 'to': 'misappropriation of Publishers’ content', 'relationship': 'ACCUSED_OF'}, {'from': 'Axel Springer', 'to': 'deal to license news for AI
model training', 'relationship': 'PARTICIPATED_IN'}, {'from': 'OpenAI', 'to': 'deal to license news for AI model training', 'relationship': 'PARTICIPATED_IN'},
{'from': 'app store antitrust issues', 'to': 'Epic Games lawsuit against Google over app store antitrust issues', 'relationship': 'MENTIONS'}, {'from': 'Epic
Games', 'to': 'Epic Games lawsuit against Google over app store antitrust issues', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Epic Games
lawsuit against Google over app store antitrust issues', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Epic Games lawsuit against Google over
app store antitrust issues', 'relationship': 'ACCUSED_OF'}, {'from': 'Epic Games lawsuit against Google over app store antitrust issues', 'to': 'app store
antitrust issues', 'relationship': 'MENTIONS'}, {'from': 'Alphabet', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms
to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Helena World Chronicle', 'to': 'News publisher files class action antitrust suit against Google,
citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'U.S. District Court in D.C.', 'to': 'News publisher files class action antitrust
suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Search Generative Experience', 'to': 'News publisher files
class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Bard AI chatbot', 'to': 'News
publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Knowledge Graph',
'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from':
'anticompetitive behavior', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship':
'MENTIONS'}, {'from': 'U.S. antitrust law', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line',
'relationship': 'MENTIONS'}, {'from': 'Sherman Act', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom
line', 'relationship': 'MENTIONS'}, {'from': 'Arkansas', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their
bottom line', 'relationship': 'MENTIONS'}, {'from': 'question-and-answer technologies', 'to': 'News publisher files class action antitrust suit against Google,
citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Publishers', 'to': 'News publisher files class action antitrust suit against
Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Knowledge Panel', 'to': 'News publisher files class action antitrust
suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Featured Snippets', 'to': 'News publisher files class
action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'class action antitrust suit', 'to': 'News
publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'misappropriation of
Publishers’ content', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship':
'MENTIONS'}, {'from': 'The Wall Street Journal', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom
line', 'relationship': 'MENTIONS'}, {'from': 'report on Thursday', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to
their bottom line', 'relationship': 'MENTIONS'}, {'from': 'deal to license news for AI model training', 'to': 'News publisher files class action antitrust suit
against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'SGE', 'to': 'News publisher files class action antitrust suit
against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Google Bard AI', 'to': 'News publisher files class action
antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'GoogleBot', 'to': 'News publisher files class
action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Google general search service', 'to':
'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'AdSense',
'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'News
Media Alliance', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'},
{'from': 'Washington Post', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship':
'MENTIONS'}, {'from': 'Epic Games lawsuit against Google over app store antitrust issues', 'to': 'News publisher files class action antitrust suit against
Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'app store antitrust issues', 'to': 'News publisher files class action
antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'U.S. Justice Department', 'to': 'News publisher
files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Hausfeld', 'to': 'News
publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Helena World
Chronicle, LLC', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'},
{'from': 'class action antitrust suit against Google', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom
line', 'relationship': 'MENTIONS'}, {'from': 'U.S. Justice Department lawsuit against Google', 'to': 'News publisher files class action antitrust suit against
Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': '2020 Justice Department civil antitrust suit over search and search
advertising', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'},
{'from': 'digital ad technologies', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line',
'relationship': 'MENTIONS'}, {'from': 'Clayton Act', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom
line', 'relationship': 'MENTIONS'}, {'from': 'search and search advertising', 'to': 'News publisher files class action antitrust suit against Google, citing
AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'digital news and reference publishing', 'to': 'News publisher files class action
antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'class-wide monetary and injunctive relief', 'to':
'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'free
marketplace of ideas', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship':
'MENTIONS'}, {'from': 'free press', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line',
'relationship': 'MENTIONS'}, {'from': 'Epic Games', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom
line', 'relationship': 'MENTIONS'}, {'from': 'antitrust', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their
bottom line', 'relationship': 'MENTIONS'}, {'from': 'The Atlantic', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to
their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Axel Springer', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s
harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Allen Institute for AI', 'to': 'News publisher files class action antitrust suit against
Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'class action lawsuit', 'to': 'News publisher files class action
antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'consumers', 'to': 'News publisher files class
action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'AI', 'to': 'News publisher files class
action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'AI technologies', 'to': 'News publisher
files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Similarweb', 'to': 'News
publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'News
publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'}, {'from': 'artificial
intelligence', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'},
{'from': 'D.C.', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'},
{'from': 'OpenAI', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'},
{'from': 'labor', 'to': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line', 'relationship': 'MENTIONS'},
{'from': 'U.S. Justice Department lawsuit against Google', 'to': 'U.S. Justice Department', 'relationship': 'PARTICIPATED_IN'}, {'from': 'class action antitrust
suit against Google', 'to': 'Helena World Chronicle, LLC', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Helena World Chronicle, LLC', 'to': 'class action
antitrust suit against Google', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'class action antitrust suit against Google', 'relationship':
'ACCUSED_OF'}, {'from': 'U.S. Justice Department', 'to': 'U.S. Justice Department lawsuit against Google', 'relationship': 'PARTICIPATED_IN'}, {'from':
'Google', 'to': 'digital ad technologies', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'There’s something going on with AI startups in France',
'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'User Choice Billing', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'Google first-party billing
system', 'relationship': 'MADE'}, {'from': 'Epic Games', 'to': 'Epic Games’ antitrust battle with Google', 'relationship': 'PARTICIPATED_IN'}, {'from':
'Google', 'to': 'Epic Games’ antitrust battle with Google', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Google Play', 'relationship': 'MADE'},
{'from': 'Google', 'to': 'Google Play', 'relationship': 'OWNS'}, {'from': 'Epic Games', 'to': 'App store competition', 'relationship': 'SAID'}, {'from':
'Google', 'to': 'App store competition', 'relationship': 'SAID'}, {'from': 'U.S. antitrust law', 'to': 'What to know about Fortnite maker Epic Games’ antitrust
battle with Google, starting today', 'relationship': 'MENTIONS'}, {'from': 'Sherman Act', 'to': 'What to know about Fortnite maker Epic Games’ antitrust battle
with Google, starting today', 'relationship': 'MENTIONS'}, {'from': 'Epic Games', 'to': 'What to know about Fortnite maker Epic Games’ antitrust battle with
Google, starting today', 'relationship': 'MENTIONS'}, {'from': 'antitrust', 'to': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google,
starting today', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting
today', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Google Play Store', 'relationship': 'OWNS'}, {'from': 'Epic Games', 'to': 'agreement with
OnePlus', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Epic Games', 'to': 'deal between Google and Samsung', 'relationship': 'SAID'}, {'from': 'Google', 'to':
'deal between Google and Samsung', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Google developer agreements', 'relationship':
'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'anti-competitive practices', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'three-year agreement',
'relationship': 'PARTICIPATED_IN'}, {'from': 'Epic Games lawsuit against Google over app store antitrust issues', 'to': 'Epic Games', 'relationship':
'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Epic Games', 'relationship': 'SAID'}, {'from': 'Epic Games', 'to': 'Sideloading', 'relationship': 'SAID'},
{'from': 'Epic Games', 'to': 'Off-the-record chats', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Chat logs and documents provided to the court',
'relationship': 'SAID'}, {'from': 'Epic Games', 'to': 'Play Store access and payment', 'relationship': 'SAID'}, {'from': 'Epic Games', 'to': "Epic's antitrust
lawsuit", 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': "Epic's antitrust lawsuit", 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to':
"Google's antitrust lawsuit with the Department of Justice", 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'search monopoly', 'relationship':
'ACCUSED_OF'}, {'from': 'Google', 'to': 'antitrust', 'relationship': 'ACCUSED_OF'}, {'from': 'Epic Games', 'to': 'antitrust battle with Google', 'relationship':
'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'antitrust battle with Google', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Epic Games', 'to': 'antitrust battle
with Apple', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Founders, are events useful?', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to':
'Gradient Ventures', 'relationship': 'PART_OF'}, {'from': 'Google', 'to': 'Passwords? We don’t need no steenkin’ passwords', 'relationship': 'MENTIONS'},
{'from': 'Google', 'to': 'Search', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'One year later, ChatGPT is still alive and kicking', 'relationship':
'MENTIONS'}, {'from': 'deal to license news for AI model training', 'to': 'Axel Springer', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Axel Springer', 'to':
'real-time ChatGPT news deal', 'relationship': 'PARTICIPATED_IN'}, {'from': 'OpenAI', 'to': 'real-time ChatGPT news deal', 'relationship': 'PARTICIPATED_IN'},
{'from': 'Google', 'to': 'ChatGPT', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to': 'ChatGPT: Everything you need to know about the AI-powered
chatbot', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Bing', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to': '2023 I/O event',
'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'DeepMind', 'relationship': 'OWNS'}, {'from': 'Google', 'to': 'DeepMind', 'relationship':
'PART_OF'}, {'from': 'Epic Games', 'to': 'A comprehensive list of 2023 tech layoffs', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'A comprehensive
list of 2023 tech layoffs', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Google Users & Products', 'relationship': 'PART_OF'}, {'from': 'Google',
'to': 'workforce reductions', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Area 120', 'relationship': 'PART_OF'}, {'from': 'Epic Games', 'to': 'Video
games in 2023: Acquisitions, layoffs, unions', 'relationship': 'MENTIONS'}, {'from': 'Epic Games', 'to': 'Mediatonic', 'relationship': 'OWNS'}, {'from':
'Google', 'to': 'How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'open source AI',
'relationship': 'SAID'}, {'from': 'Google', 'to': 'Google Pixel Buds Pro', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'Google Pixel Watch',
'relationship': 'MADE'}, {'from': 'Google', 'to': 'Govee’s colorful LED light strips are on sale just in time for Halloween', 'relationship': 'MENTIONS'},
{'from': 'Google', 'to': 'From Ten Hag’s future to a leaking roof: Ratcliffe’s Manchester United in-tray', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to':
'The best Black Friday deals you can get for under $25', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Apple', 'relationship': 'MENTIONS'}, {'from':
'Google', 'to': 'Apple', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Apple', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to': 'Everton players
and fans will be fired up by points penalty, asserts Sean Dyche', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Discogs’ vibrant vinyl community is
shattering', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Biden’s AI EO hailed as broad, but not deep without legislation to match', 'relationship':
'MENTIONS'}, {'from': 'Google', 'to': 'Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn', 'relationship': 'MENTIONS'}, {'from': 'Google',
'to': 'Why We Desperately Need a New (and Better) Google', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Reddit', 'relationship': 'COMPETES_WITH'},
{'from': 'Google', 'to': 'large language models', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to': 'federal government', 'relationship': 'SAID'},
{'from': 'Google', 'to': 'monopoly', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'Forrester', 'relationship': 'SAID'}, {'from': 'Google', 'to':
'Matt Cutts', 'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'web spam team', 'relationship': 'PART_OF'}, {'from': 'Google', 'to': 'Danny Sullivan',
'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'Sullivan', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Sullivan', 'relationship': 'WORKS_FOR'},
{'from': 'Google', 'to': 'The people who ruined the internet', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'SEOs', 'relationship': 'SAID'}, {'from':
'Google', 'to': 'Ray', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Cutts', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Cutts', 'relationship':
'WORKS_FOR'}, {'from': 'Google', 'to': 'Lee', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'People are turning to Snap Map for firsthand perspectives from
Gaza', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'misinformation', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Early impressions of Google’s
Gemini aren’t great', 'relationship': 'MENTIONS'}, {'from': 'OpenAI', 'to': 'Early impressions of Google’s Gemini aren’t great', 'relationship': 'MENTIONS'},
{'from': 'Google', 'to': 'Bard', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Bard', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'Bard',
'relationship': 'OWNS'}, {'from': 'Google', 'to': 'Gemini Pro', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Gemini Pro', 'relationship': 'MADE'},
{'from': 'Google', 'to': 'The restaurant nearest Google', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Thai food near me', 'relationship': 'SAID'},
{'from': 'Google', 'to': 'local search', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Meta seeks legislation that would require parents to approve teens’
app downloads', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Sacha Haworth', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'bias', 'relationship':
'SAID'}, {'from': 'Google', 'to': 'anti-competition law', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'Riot', 'relationship': 'OWNS'}, {'from':
'Epic Games', 'to': 'Tim Sweeney on Epic’s victory royale over Google', 'relationship': 'MENTIONS'}, {'from': 'antitrust', 'to': 'Tim Sweeney on Epic’s victory
royale over Google', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Tim Sweeney on Epic’s victory royale over Google', 'relationship': 'MENTIONS'},
{'from': 'Google', 'to': 'Activision competing store', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Activision-Google deal', 'relationship':
'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Epic’s victory royale over Google', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Epic Games', 'to': 'Tim
Sweeney', 'relationship': 'WORKS_FOR'}, {'from': 'Epic Games', 'to': 'Tim Sweeney', 'relationship': 'FOUNDED'}, {'from': 'Google', 'to': 'Tim Sweeney',
'relationship': 'SAID'}, {'from': 'Google', 'to': 'Don Harrison', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Epic v. Google', 'relationship':
'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Epic', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Epic', 'relationship': 'ACCUSED_OF'}, {'from': 'Google',
'to': 'Epic', 'relationship': 'COMPETES_WITH'}, {'from': 'Epic Games', 'to': 'Epic Games Store', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'England
thrash Scotland but cannot avoid Team GB Olympic heartbreak', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': '‘Mass breach of privacy’: TikTok under fire
for tracking users online', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Samantha Hoffman', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'US',
'relationship': 'LOCATED_IN'}, {'from': 'Google', 'to': 'If your startup doesn’t seem impossible at first, it’s not hard enough', 'relationship': 'MENTIONS'},
{'from': 'Google', 'to': 'Google Maps', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'Google Maps', 'relationship': 'OWNS'}, {'from': 'Google', 'to': 'The
best Prime Day smart home deals', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'England choose survival over oblivion and a chance of summer trip to
France', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more',
'relationship': 'MENTIONS'}, {'from': 'Google', 'to': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court",
'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'The best VPN services for 2023', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Google One VPN',
'relationship': 'MADE'}, {'from': 'Google', 'to': 'Wilkins', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Keep Labs’ smart storage redefines medication
management', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'The best October Prime Day gaming deals', 'relationship': 'MENTIONS'}, {'from': 'Google',
'to': 'U.S. government', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'U.S. government’s antitrust trial against Google', 'relationship':
'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Prabhakar Raghavan', 'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'Pichai', 'relationship': 'SAID'},
{'from': 'Google', 'to': 'lucrative search agreement', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'systematically destroying chats',
'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'David Drummond', 'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'Brave', 'relationship':
'COMPETES_WITH'}, {'from': 'Google', 'to': 'anticompetitive conduct', 'relationship': 'ACCUSED_OF'}, {'from': 'U.S. Justice Department', 'to': '5 things we
learned so far about the Google antitrust case', 'relationship': 'MENTIONS'}, {'from': 'antitrust', 'to': '5 things we learned so far about the Google antitrust
case', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': '5 things we learned so far about the Google antitrust case', 'relationship': 'MENTIONS'}, {'from':
'Google', 'to': 'DOJ', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'antitrust law', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'Ten Hag’s
gamble pays off against Chelsea as McTominay steals show | Jonathan Liew', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Deepfake election risks
trigger EU call for more generative AI safeguards', 'relationship': 'MENTIONS'}, {'from': 'Alphabet', 'to': 'Google', 'relationship': 'OWNS'}, {'from':
'Alphabet', 'to': 'Google', 'relationship': 'PART_OF'}, {'from': 'Helena World Chronicle', 'to': 'Google', 'relationship': 'SAID'}, {'from': 'anticompetitive
behavior', 'to': 'Google', 'relationship': 'ACCUSED_OF'}, {'from': 'U.S. antitrust law', 'to': 'Google', 'relationship': 'ACCUSED_OF'}, {'from': 'Sherman Act',
'to': 'Google', 'relationship': 'ACCUSED_OF'}, {'from': 'class action antitrust suit', 'to': 'Google', 'relationship': 'ACCUSED_OF'}, {'from': 'misappropriation
of Publishers’ content', 'to': 'Google', 'relationship': 'ACCUSED_OF'}, {'from': 'Epic Games lawsuit against Google over app store antitrust issues', 'to':
'Google', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Epic Games lawsuit against Google over app store antitrust issues', 'to': 'Google', 'relationship':
'ACCUSED_OF'}, {'from': 'class action antitrust suit against Google', 'to': 'Google', 'relationship': 'ACCUSED_OF'}, {'from': 'digital ad technologies', 'to':
'Google', 'relationship': 'ACCUSED_OF'}, {'from': 'Epic Games', 'to': 'Google', 'relationship': 'SAID'}, {'from': 'antitrust', 'to': 'Google', 'relationship':
'ACCUSED_OF'}, {'from': 'OpenAI', 'to': 'Google', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to': 'AI Principles', 'relationship': 'MENTIONS'},
{'from': 'Google', 'to': 'ethical reviews', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'security', 'relationship': 'SAID'}, {'from': 'Google', 'to':
'safety', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Google-developed AI tools', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'The 30 best Amazon
Prime Day deals for October 2023', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Partnership on AI', 'relationship': 'PART_OF'}, {'from': 'Google',
'to': 'Meta’s open source AI approach', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'IBM', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to':
'enterprise-focused AI services', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'Meta and IBM form an AI Alliance, but to what end?', 'relationship':
'MENTIONS'}, {'from': 'Google', 'to': 'NMPA: Generative AI is ‘the greatest risk to the human creative class that has ever existed’', 'relationship':
'MENTIONS'}, {'from': 'Google', 'to': 'Talks to secure licenses for artists’ vocals and musical melodies', 'relationship': 'PARTICIPATED_IN'}, {'from':
'Google', 'to': 'AI training', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'trial', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Epic Games', 'to': 'The
Epic v. Google trial may come down to simple v. complicated', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'The Epic v. Google trial may come down to
simple v. complicated', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Gary Bornstein', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'unlawful
monopoly', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'Bornstein', 'relationship': 'SAID'}, {'from': 'Epic Games', 'to': 'Epic sued Google',
'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Epic sued Google', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Glenn Pomerantz',
'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'Pomerantz', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Android sideloading flow',
'relationship': 'SAID'}, {'from': 'Google', 'to': 'making potential evidence disappear', 'relationship': 'ACCUSED_OF'}, {'from': 'Epic Games', 'to': 'Epic v.
Google trial', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Epic v. Google trial', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to':
'Deft', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to': 'Zach Hudson', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'multi-search',
'relationship': 'MADE'}, {'from': 'Google', 'to': 'Deft is building e-commerce search to help you find the right product quickly', 'relationship': 'MENTIONS'},
{'from': 'Google', 'to': 'multi-search feature', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'AI-generated content', 'relationship': 'SAID'}, {'from':
'Google', 'to': "Sweeping White House AI executive order takes aim at the technology's toughest challenges", 'relationship': 'MENTIONS'}, {'from': 'Google',
'to': 'These Are Our Favorite Walmart Cyber Monday Deals', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Nest Cam', 'relationship': 'MADE'}, {'from':
'Google', 'to': 'Ten Hag demands both positivity and pragmatism from erratic Onana', 'relationship': 'MENTIONS'}, {'from': 'deal to license news for AI model
training', 'to': 'OpenAI', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'OpenAI', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to':
'Ilya Sutskever', 'relationship': 'WORKS_FOR'}, {'from': 'AI', 'to': 'Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs',
'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs', 'relationship':
'MENTIONS'}, {'from': 'OpenAI', 'to': 'Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs', 'relationship': 'MENTIONS'}, {'from':
'Google', 'to': 'Gemini', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Gemini', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'Gemini',
'relationship': 'OWNS'}, {'from': 'Google', 'to': 'Gemini Ultra', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'Earnings week ahead: Amazon, Microsoft,
Alphabet, Meta, Ford, GM and more', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'European roundup: Bellingham helps Real Madrid go top while Bayern
draw', 'relationship': 'MENTIONS'}, {'from': 'Epic Games', 'to': 'Epic v. Apple trial', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'illegal
tying', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'Android app distribution', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'Android
in-app payment processing', 'relationship': 'ACCUSED_OF'}, {'from': 'Google', 'to': 'sideloaded apps equal gaping security holes', 'relationship': 'SAID'},
{'from': 'Google', 'to': 'tentative settlement over app store antitrust claims', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'settlement',
'relationship': 'PARTICIPATED_IN'}, {'from': 'Epic Games', 'to': 'Epic v. Google, explained', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Epic v.
Google, explained', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'deleting potentially incriminating messages', 'relationship': 'ACCUSED_OF'}, {'from':
'Google', 'to': 'US v. Google antitrust trial', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': "Epic's trial", 'relationship': 'PARTICIPATED_IN'},
{'from': 'Google', 'to': 'Epic v. Google lawsuit', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Publishers', 'to': 'Europe’s artists and labels could lose
$137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'TikTok',
'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to': 'The Ray-Ban Meta smart glasses actually make the future look cool', 'relationship': 'MENTIONS'},
{'from': 'Google', 'to': '8 predictions for AI in 2024', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Here are the best deals on headphones and
wireless earbuds we could find', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Google Pixel Buds A-Series', 'relationship': 'MADE'}, {'from': 'Google',
'to': 'Ten Hag’s Manchester United stand up and die at club stuck in the past tense', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Justice
Department', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis
Kooker’s speech about AI', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'fair use', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Google’s
submission to the United States Copyright Office', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Google’s recent filing with the US Copyright
Office', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment',
'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'YouTube', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'YouTube', 'relationship': 'OWNS'}, {'from':
'Google', 'to': 'Alphabet Inc.', 'relationship': 'PART_OF'}, {'from': 'Google', 'to': 'Google filing with the US Copyright Office', 'relationship':
'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Artificial Intelligence and Copyright', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'ingestion of
copyrighted material for training generative AI platforms', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'Google USCO submission', 'relationship':
'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment.', 'relationship': 'MENTIONS'},
{'from': 'Google', 'to': 'US copyright law', 'relationship': 'SAID'}, {'from': 'Google', 'to': 'copyright harvesting', 'relationship': 'SAID'}, {'from':
'Google', 'to': 'Google’s USCO filing', 'relationship': 'PARTICIPATED_IN'}, {'from': 'Google', 'to': 'Sony Music’s artists aren’t involved in YouTube’s new
voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office.', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Temu
was the most-downloaded iPhone app in the U.S. in 2023', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'privacy', 'relationship': 'SAID'}, {'from':
'Google', 'to': 'Know the price-matching policies for Best Buy, Target, Walmart, and others', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'The best
Black Friday headphone deals from Sony, Apple, Bose, Beats and more', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Israel', 'relationship':
'LOCATED_IN'}, {'from': 'Google', 'to': 'Israel’s startup ecosystem: down but not out', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Warner Music
Group boss Robert Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference', 'relationship': 'MENTIONS'}, {'from':
'Google', 'to': 'French soccer players back African mobile video network StarNews Mobile in $3M round', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to':
'EU commissioner sidesteps MEPs’ questions about CSAM microtargeting', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'The new, ‘efficient’ Spotify has a
very different approach to podcasting', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'ASX set to rise ahead of inflation report after big tech lifts
Wall Street', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Google Cloud Platform', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'Spill toasts one
year with a 2.7M seed extension, Kerry Washington and champagne', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'The UK just laid out new rules for the
internet — it only gets harder from here', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Lewis', 'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to':
'Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to':
'TicWatch Pro 5', 'relationship': 'COMPETES_WITH'}, {'from': 'Google', 'to': '78 Absolute Best October Prime Day Deals (2023)', 'relationship': 'MENTIONS'},
{'from': 'Google', 'to': 'EU says incoming rules for general purpose AIs can evolve over time', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to':
'generative AI', 'relationship': 'SAID'}, {'from': 'Google', 'to': "India to be one of world's fastest growing economies over next decade", 'relationship':
'MENTIONS'}, {'from': 'Google', 'to': 'Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling', 'relationship': 'MENTIONS'}, {'from':
'Google', 'to': 'Liverpool enjoy home comforts but Dyche’s Everton offer derby danger | Andy Hunter', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to':
'Google reCaptcha', 'relationship': 'MADE'}, {'from': 'Epic Games', 'to': 'Can anyone survive Fortnite as a job?', 'relationship': 'MENTIONS'}, {'from': 'Epic
Games', 'to': 'Fortnite', 'relationship': 'MADE'}, {'from': 'Epic Games', 'to': 'Fortnite', 'relationship': 'OWNS'}, {'from': 'Google', 'to': 'The 26 best
Amazon Prime Day deals you can still shop for today', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Google Nest Thermostat', 'relationship': 'MADE'},
{'from': 'Google', 'to': 'Some of our favorite devices are on sale for Black Friday.', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Google Nest Hub
(second-gen)', 'relationship': 'MADE'}, {'from': 'Google', 'to': 'The best anti-Prime Big Deal Day deals', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to':
'Pebble, the Twitter alternative previously known as T2, is shutting down', 'relationship': 'MENTIONS'}, {'from': 'Google', 'to': 'Bradley Horowitz',
'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'The best Prime Day deals on headphones for October 2023', 'relationship': 'MENTIONS'}, {'from':
'Google', 'to': 'Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas', 'relationship': 'MENTIONS'}, {'from': 'Google',
'to': 'Dalia Awad', 'relationship': 'WORKS_FOR'}, {'from': 'Google', 'to': 'The best apps and tools for managing your money online', 'relationship':
'MENTIONS'}]
</item 4>

<item 5>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Project Hug
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Project Hug
- Tim Sweeney -SAID-> Project Hug
- Project Hug -PART_OF-> Google Games Velocity Program
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Epic

Entities: Project Hug, Epic, Google, Epic v. Google, explained, Sweeney, state attorneys general, Project Agave, Project Banyan, game developers, phone makers,
Google's app store, settlement, anticompetitive practices, paying game developers and phone makers not to abandon Google's app store, deleting potentially
incriminating messages
</item 5>

<item 6>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sundar Pichai
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> U.S. antitrust law
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sherman Act
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> User Choice Billing
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Google first-party billing system

Entities: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, antitrust, antitrust battle with Google,
antitrust battle with Apple, Apple, Supreme Court, TechCrunch, Google, Fortnite
</item 6>

<item 7>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Android

Entities: Epic Games’ antitrust battle with Google, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic
Games, Sideloading, Security risks, Off-the-record chats, Chat logs and documents provided to the court, Play Store access and payment, TechCrunch, Google,
competition, Play Store, Fortnite
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Epic Games
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Epic Games
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Epic Games
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Epic Games
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Epic Games

Entities: Epic Games, Sean, Apple, Google, Epic v. Google trial, Epic v. Apple trial, #FreeFortnite, Llamacorn, antitrust claims, illegal monopoly, Epic v.
Google, explained, Fortnite, The Verge
</item 8>

<item 9>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> User Choice Billing
- The fate of US venture capital in China teeters on uncertainty -MENTIONS-> Tencent
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Tencent
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Tencent
- European neobroker Scalable Capital raises $65M on a flat $1.4B valuation -MENTIONS-> Tencent

Entities: User Choice Billing, Tencent, Epic, China, Google, Epic v. Google, explained, Epic v. Google lawsuit, Judge Donato, Samsung Galaxy phone, The Verge
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=07dad088ed1419d5 ===
