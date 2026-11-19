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

=== ITEM id=80b2fa8f3a6f5d68 ===
Gold sentences and facts:
Sentence 0: Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
  Fact 0.0: Google unveiled new hardware.
  Fact 0.1: This week was Google's annual hardware event.
  Fact 0.2: Google is a search and consumer tech giant.
  Fact 0.3: At the hardware event, Google showed off what Google had been working on.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: The last roundup reported how Google spent $26.3 billion in 2021 making Google the default search engine across platforms.
  Fact 1.1: The last roundup reported how Google tried to have Chrome preinstalled on iPhones.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content.
  Fact 2.2: The case argues that Google "siphons off" news publishers' readers.
  Fact 2.3: The case argues that Google "siphons off" news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.

Graph facts:
- Grey -WORKS_FOR-> DuckDuckGo
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> DuckDuckGo
- Is Google Search better than the rest? And is that fair? -MENTIONS-> DuckDuckGo
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> third-party developers
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> antitrust

Entities: third-party developers, DuckDuckGo, antitrust, Bing, Microsoft, Apple, monopoly, TechCrunch, data, Google antitrust case, Brave, Microsoft trial,
anticompetitive conduct, APIs, computer manufacturers
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
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Bellshaw
- Microsoft -MENTIONS-> cloud gaming
- Microsoft -MENTIONS-> device-agnostic ecosystem
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Microsoft
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Microsoft

Entities: Bellshaw, Microsoft, TechCrunch, Pichai, Google antitrust case, message exchange, litigation hold, personnel matter, Internet Explorer, David
Drummond, Internet Explorer became the default search engine, Drummond’s letter, default settings, 5 things we learned so far about the Google antitrust case,
Google
</item 3>

<item 4>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Meagan Bellshaw
- U.S. Justice Department -PARTICIPATED_IN-> U.S. Justice Department lawsuit against Google
- Meagan Bellshaw -WORKS_FOR-> Justice Department
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. Justice Department
- U.S. Justice Department -PARTICIPATED_IN-> 2020 Justice Department civil antitrust suit over search and search advertising

Entities: Meagan Bellshaw, U.S. Justice Department, TechCrunch, Pichai, Nitin, homepage market share, search market share, Google antitrust case, 5 things we
learned so far about the Google antitrust case, Google, The Verge
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.

Graph facts:
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Chrome
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Chrome
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Chrome
- ‘Mass breach of privacy’: TikTok under fire for tracking users online -MENTIONS-> Chrome
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Chrome

Entities: Chrome, Mozilla, Apple, TechCrunch, Prabhakar Raghavan, Pichai, trial hearing, court testimony, Google antitrust case, decision made with the consumer
in mind, 5 things we learned so far about the Google antitrust case, Samsung, Google, New York Times, iPhones
</item 7>

<item 8>
Graph query row:
article: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
published_on: 2023-12-15T17:56:02+00:00
connected_entities: [{'name': 'Alphabet', 'relationship': 'MENTIONS'}, {'name': 'Helena World Chronicle', 'relationship': 'MENTIONS'}, {'name': 'U.S. District
Court in D.C.', 'relationship': 'MENTIONS'}, {'name': 'Search Generative Experience', 'relationship': 'MENTIONS'}, {'name': 'Bard AI chatbot', 'relationship':
'MENTIONS'}, {'name': 'Knowledge Graph', 'relationship': 'MENTIONS'}, {'name': 'anticompetitive behavior', 'relationship': 'MENTIONS'}, {'name': 'U.S. antitrust
law', 'relationship': 'MENTIONS'}, {'name': 'Sherman Act', 'relationship': 'MENTIONS'}, {'name': 'Arkansas', 'relationship': 'MENTIONS'}, {'name':
'question-and-answer technologies', 'relationship': 'MENTIONS'}, {'name': 'Publishers', 'relationship': 'MENTIONS'}, {'name': 'Knowledge Panel', 'relationship':
'MENTIONS'}, {'name': 'Featured Snippets', 'relationship': 'MENTIONS'}, {'name': 'class action antitrust suit', 'relationship': 'MENTIONS'}, {'name':
'misappropriation of Publishers’ content', 'relationship': 'MENTIONS'}, {'name': 'The Wall Street Journal', 'relationship': 'MENTIONS'}, {'name': 'report on
Thursday', 'relationship': 'MENTIONS'}, {'name': 'deal to license news for AI model training', 'relationship': 'MENTIONS'}, {'name': 'SGE', 'relationship':
'MENTIONS'}, {'name': 'Google Bard AI', 'relationship': 'MENTIONS'}, {'name': 'GoogleBot', 'relationship': 'MENTIONS'}, {'name': 'Google general search
service', 'relationship': 'MENTIONS'}, {'name': 'AdSense', 'relationship': 'MENTIONS'}, {'name': 'News Media Alliance', 'relationship': 'MENTIONS'}, {'name':
'Washington Post', 'relationship': 'MENTIONS'}, {'name': 'Epic Games lawsuit against Google over app store antitrust issues', 'relationship': 'MENTIONS'},
{'name': 'app store antitrust issues', 'relationship': 'MENTIONS'}, {'name': 'U.S. Justice Department', 'relationship': 'MENTIONS'}, {'name': 'Hausfeld',
'relationship': 'MENTIONS'}, {'name': 'Helena World Chronicle, LLC', 'relationship': 'MENTIONS'}, {'name': 'class action antitrust suit against Google',
'relationship': 'MENTIONS'}, {'name': 'U.S. Justice Department lawsuit against Google', 'relationship': 'MENTIONS'}, {'name': '2020 Justice Department civil
antitrust suit over search and search advertising', 'relationship': 'MENTIONS'}, {'name': 'digital ad technologies', 'relationship': 'MENTIONS'}, {'name':
'Clayton Act', 'relationship': 'MENTIONS'}, {'name': 'search and search advertising', 'relationship': 'MENTIONS'}, {'name': 'digital news and reference
publishing', 'relationship': 'MENTIONS'}, {'name': 'class-wide monetary and injunctive relief', 'relationship': 'MENTIONS'}, {'name': 'free marketplace of
ideas', 'relationship': 'MENTIONS'}, {'name': 'free press', 'relationship': 'MENTIONS'}, {'name': 'Epic Games', 'relationship': 'MENTIONS'}, {'name':
'antitrust', 'relationship': 'MENTIONS'}, {'name': 'The Atlantic', 'relationship': 'MENTIONS'}, {'name': 'Axel Springer', 'relationship': 'MENTIONS'}, {'name':
'Allen Institute for AI', 'relationship': 'MENTIONS'}, {'name': 'class action lawsuit', 'relationship': 'MENTIONS'}, {'name': 'consumers', 'relationship':
'MENTIONS'}, {'name': 'AI', 'relationship': 'MENTIONS'}, {'name': 'AI technologies', 'relationship': 'MENTIONS'}, {'name': 'Similarweb', 'relationship':
'MENTIONS'}, {'name': 'Google', 'relationship': 'MENTIONS'}, {'name': 'artificial intelligence', 'relationship': 'MENTIONS'}, {'name': 'D.C.', 'relationship':
'MENTIONS'}, {'name': 'OpenAI', 'relationship': 'MENTIONS'}, {'name': 'labor', 'relationship': 'MENTIONS'}]
article_passages: ['News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch,
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
October_7_Google_hardware_event: []
November_13_default_search_report: [{'name': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'passages': 'More
from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nMore from the US v Google
trial: Vertical search, pre-installs and the case of Firefox/Yahoo\n\nWe’re nearly two months into the Justice Department’s landmark antitrust case against
Google — one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the 1990s — and the revelations just keep getting juicier.\n\nIn
our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome
preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s most
lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.\n\nBefore
we go into some of these tidbits…\n\nWhy the Google vs. U.S.', 'published_on': '2023-11-13T14:37:44+00:00'}, {'name': 'More from the US v Google trial: vertical
search, pre-installs and the case of Firefox/Yahoo', 'passages': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo
(TechCrunch, 2023-11-13T14:37:44+00:00)\nBefore we go into some of these tidbits…\n\nWhy the Google vs. U.S. antitrust case matters\n\nThe government has argued
that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering competitors from accessing the data
they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its behavior and share its APIs with
third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer manufacturers and wireless carriers.
Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that antitrust law remains relevant and that even
though Google is basically the God of the internet, it’s still no match for the U.S. law.', 'published_on': '2023-11-13T14:37:44+00:00'}, {'name': 'More from
the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'passages': 'More from the US v Google trial: vertical search, pre-installs
and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nlaw. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC
sued Amazon in September for using anticompetitive and unfair strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for
years over the company’s policy for third-party apps on its devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and
Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.\n\nThis isn’t Google’s only antitrust case in trial right now. The search engine
giant last week settled a separate antitrust lawsuit with dating site Match Group. On November 6, Google went to trial with Fortnite maker Epic Games. The
latter hopes to prove that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission structure.\n\nNow,
onto the roundup!', 'published_on': '2023-11-13T14:37:44+00:00'}, {'name': 'More from the US v Google trial: vertical search, pre-installs and the case of
Firefox/Yahoo', 'passages': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch,
2023-11-13T14:37:44+00:00)\nNow, onto the roundup!\n\nA window into Google’s most popular search queries\n\nJudge Amit Mehta ruled to make a list public that
provides a glimpse of which search terms make Google the most money. The list of popular search terms ordered by revenue includes 20 terms that were lucrative
for the week of September 22, 2018. Information like revenue per search term, how many queries each of those terms got, along with a separate list of popular
search terms ordered by queries (not revenue), were all redacted. The list we can see is as follows:\n\niPhone 8\n\niPhone 8 plus\n\nAuto insurance\n\nCar
insurance\n\nCheap flights\n\nCar insurance quotes\n\nDirecTV\n\nOnline
colleges\n\nAT&T\n\nHulu\n\niPhone\n\nUber\n\nSpectrum\n\nComcast\n\nXfinity\n\nInsurance quotes\n\nFree credit report\n\nCheap car
insurance\n\nAARP\n\nLifeLock\n\nThere is, in reality, little surprise here.', 'published_on': '2023-11-13T14:37:44+00:00'}, {'name': 'More from the US v Google
trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'passages': 'More from the US v Google trial: vertical search, pre-installs and the case of
Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nWe’ve already established that Google and Apple have a long and mutually beneficial relationship, even
while competing, so it’s not surprising to see three Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the
official release date of the iPhone 8.\n\nMeanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to
how much Google dominates vertical search — that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot
topic in September 2017 and LifeLock was making a big push to win business with people who wanted to buy identity theft protection.\n\nRevenue-shares to
pre-install Google apps on Androids\n\nJamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8.
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.', 'published_on': '2023-11-13T14:37:44+00:00'}, {'name':
'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'passages': 'More from the US v Google trial: vertical search,
pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nHe said that the competition between Google and Apple is “as intense as it
gets,” reports Bloomberg.\n\nRosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android
smartphone makers (like Samsung or Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the
default choices, he said.\n\nGoogle also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to
set Google search and Chrome web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The
RSAs also motivate other companies to make or sell more Android devices, he said.', 'published_on': '2023-11-13T14:37:44+00:00'}, {'name': 'More from the US v
Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'passages': 'More from the US v Google trial: vertical search, pre-installs and the
case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nThe RSAs also motivate other companies to make or sell more Android devices, he said.\n\nExpedia
complains of too many ads on search, expensive ad payments\n\nOn November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the
increasing number of ads in search results having an impact on organic listings.\n\n“I must say I’m on the edge of revolt now that Google’s actions are so
punitive, not just for Expedia but also for IAC and all the players that depend upon something of a level playing field,” wrote Diller in a letter to Google
back in 2019, according to Bloomberg.\n\nGoogle CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company
built.\n\nExpedia execs also testified about the cost of ads and how increases had no impact on search results.', 'published_on': '2023-11-13T14:37:44+00:00'},
{'name': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'passages': 'More from the US v Google trial: vertical
search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)\nOn October 19, Expedia’s former chief operating officer, Jeff Hurst,
told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290 million in 2019. And yet, Expedia’s traffic from Google did not
increase. The implication was that this was due to direct competition from Google itself. Hurst pointed out that Google began sharing its own flight and hotel
data in search results in that period, according to The Seattle Times.\n\nEuropean antitrust fine spurs Google to beef up search engine quality\n\nThe
government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust fine
in 2018, internal documents revealed, according to Bloomberg.\n\nThe EU’s antitrust order forced Google to offer Android phone users a screen with five search
engine options to choose from, according to the DOJ.', 'published_on': '2023-11-13T14:37:44+00:00'}, {'name': 'More from the US v Google trial: vertical search,
pre-installs and the case of Firefox/Yahoo', 'passages': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo
(TechCrunch, 2023-11-13T14:37:44+00:00)\nIn response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and
Germany in 2019 and 2020 with more local content: news, post-game soccer video highlights, information on local television options for streaming, and
pronunciation practice for different language. The aim was to incentivize users to click on Google’s home screen, rather than the competition’s.\n\nThat
revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its products, a
classic consequence in a monopoly.\n\nMozilla digs in on Google’s supremacy in search\n\nInterestingly, on November 1 Mozilla CEO Mitchell Baker provided a
defense of Google’s quality as a search engine, even in “competitive” environments. Specifically, Baker recounted how, Mozilla “failed” when it switched
Firefox’s default search engine from Google to Yahoo.', 'published_on': '2023-11-13T14:37:44+00:00'}, {'name': 'More from the US v Google trial: vertical
search, pre-installs and the case of Firefox/Yahoo', 'passages': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo
(TechCrunch, 2023-11-13T14:37:44+00:00)\nQuick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be
the default search engine on Firefox. At the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.\n\n“I felt strongly that Yahoo was
not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said Yahoo had promised to reduce the
number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.\n\n“The number of users who stayed with Firefox declined
noticeably during the years when Yahoo was the default,” said Baker.\n\nBaker, who was providing a recorded deposition for Google’s defense, noted that
Mozilla’s users apparently wanted and expected Google.\n\nHowever, that is not the full story. Yahoo was already well behind Google in terms of search
technology at that point.', 'published_on': '2023-11-13T14:37:44+00:00'}, {'name': 'More from the US v Google trial: vertical search, pre-installs and the case
of Firefox/Yahoo', 'passages': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch,
2023-11-13T14:37:44+00:00)\nYahoo was already well behind Google in terms of search technology at that point. But also, Firefox was very well behind Chrome,
which ended 2014 with nearly 50% market share and ended 2017 (when Google got prime default search position on Firefox) at nearly 65% of all web browser usage
on desktop: and mobile is even more skewed to Chrome. In other words, the number of Firefox users might have been declining for other reasons, although pushing
the blame on Yahoo certainly works in Google’s favor here!\n\nThe trial continues…', 'published_on': '2023-11-13T14:37:44+00:00'}]
</item 8>

<item 9>
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
</item 9>

<item 10>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified
that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO
Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.

Google agreed not to promote Chrome to Safari users

As part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with
banners, pop-ups and other annoying means in other Google apps.

The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.

Google tried to be preinstalled on iPhones

Pichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018, according to The Verge.

Graph facts:
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Chrome
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Chrome
- Arteta reveals Pochettino influence on his career | Poch: He is family -MENTIONS-> Chrome
- ‘Mass breach of privacy’: TikTok under fire for tracking users online -MENTIONS-> Chrome
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Chrome

Entities: Chrome, Satya Nadella, DuckDuckGo, Bing, The Information, Microsoft, Apple, TechCrunch, Tim Cook, Pichai, Google antitrust case, 5 things we learned
so far about the Google antitrust case, Google, iOS, iPhone
</item 10>
=== END ITEM id=80b2fa8f3a6f5d68 ===

=== ITEM id=f8403deb02faec03 ===
Gold sentences and facts:
Sentence 0: This whole line of argument is called “market definition.” If the court decides the relevant market is phones and app stores, not specifically Android ones, then Google’s on the road to victory.
  Fact 0.0: The whole line of argument is called 'market definition.'
  Fact 0.1: If the court decides the relevant market is phones and app stores, rather than specifically Android ones, Google is on the road to victory.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: The woman admits that she may have misinterpreted the evidence.
  Fact 1.1: The woman maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 1>

<item 2>
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
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Is Google Search better than the rest? And is that fair?, Grey, DuckDuckGo, privacy invasion, inappropriate data handling, organic
search results, ad systems affecting organic results, Danny Sullivan, X, The Age, Google, WIRED, ad revenue
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- Alphabet -ACCUSED_OF-> Sherman Act
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Sherman Act
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.

Entities: Sherman Act, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, U.S. Justice Department,
Hausfeld, Helena World Chronicle, LLC, class action antitrust suit against Google, U.S. Justice Department lawsuit against Google, 2020 Justice Department civil
antitrust suit over search and search advertising, digital ad technologies, Clayton Act, search and search advertising, digital news and reference publishing,
class-wide monetary and injunctive relief, free marketplace of ideas, free press
</item 5>

<item 6>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Wired opinion article, Megan Gray, Is Google Search better than the rest? And is that fair?, Walker, The Age, US, Google, trial, WIRED
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.

Graph facts:
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal
- Europe’s artists and labels could lose $137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn -MENTIONS-> Publishers
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet

Entities: Knowledge Graph, Publishers, Knowledge Panel, Featured Snippets, class action antitrust suit, misappropriation of Publishers’ content, The Wall Street
Journal, report on Thursday, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, antitrust, TechCrunch, AI,
Google
</item 7>

<item 8>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Early impressions of Google’s Gemini aren’t great -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, Is Google Search better than the rest? And is that fair?, Megan Grey, semantic keyword matching, The Age, Google
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- report on Thursday -PUBLISHED_BY-> The Wall Street Journal
- Publishers -PARTICIPATED_IN-> class action antitrust suit
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal

Entities: class action antitrust suit, The Wall Street Journal, deal to license news for AI model training, News publisher files class action antitrust suit
against Google, citing AI’s harms to their bottom line, antitrust, The Atlantic, Axel Springer, TechCrunch, AI, Similarweb, Google, OpenAI
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=f8403deb02faec03 ===
