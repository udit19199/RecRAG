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

=== ITEM id=8655a53bf143f722 ===
Gold sentences and facts:
Sentence 0: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 0.0: Epic will also suggest that Google hid items from discovery by allowing executives and employees to have 'off the record' chats.
  Fact 0.1: Google will counter this, saying it has provided the court with thousands of chat logs.
  Fact 0.2: Google says it has provided the court with millions of documents.
Sentence 1: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 1.0: Google did not release the full model, Gemini Ultra.
  Fact 1.1: Google released only a 'lite' version called Gemini Pro.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that this happens through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
Graph query row:
article: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
published_on: 2023-12-15T17:56:02+00:00
connected_entities: ['TechCrunch', 'Alphabet', 'Helena World Chronicle', 'U.S. District Court in D.C.', 'Search Generative Experience', 'Bard AI chatbot',
'Knowledge Graph', 'anticompetitive behavior', 'U.S. antitrust law', 'Sherman Act', 'Arkansas', 'question-and-answer technologies', 'Publishers', 'Knowledge
Panel', 'Featured Snippets', 'class action antitrust suit', 'misappropriation of Publishers’ content', 'The Wall Street Journal', 'report on Thursday', 'deal to
license news for AI model training', 'SGE', 'Google Bard AI', 'GoogleBot', 'Google general search service', 'AdSense', 'News Media Alliance', 'Washington Post',
'Epic Games lawsuit against Google over app store antitrust issues', 'app store antitrust issues', 'U.S. Justice Department', 'Hausfeld', 'Helena World
Chronicle, LLC', 'class action antitrust suit against Google', 'U.S. Justice Department lawsuit against Google', '2020 Justice Department civil antitrust suit
over search and search advertising', 'digital ad technologies', 'Clayton Act', 'search and search advertising', 'digital news and reference publishing',
'class-wide monetary and injunctive relief', 'free marketplace of ideas', 'free press', 'Epic Games', 'antitrust', 'The Atlantic', 'Axel Springer', 'Allen
Institute for AI', 'class action lawsuit', 'consumers', 'AI', 'AI technologies', 'Similarweb', 'Google', 'artificial intelligence', 'D.C.', 'OpenAI', 'labor']
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
line of commerce.”', 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch,
2023-12-15T17:56:02+00:00)\nSGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s
“walled garden” as it “plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service,
GoogleBot.\n\nPlus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from
the News Media Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for
AI, had found that News and Media sites were the third largest category of AI training data.)\n\nThe case points to other concerns, too, like changing AdSense
rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat messages — an issue raised in the recent Epic Games lawsuit
against Google over app store antitrust issues, which Epic won.', 'News publisher files class action antitrust suit against Google, citing AI’s harms to their
bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)\nThe case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for
monopolizing digital ad technologies, and references the 2020 Justice Department’s civil antitrust suit over search and search advertising (which are different
markets from digital ad technologies in the more recent suit).\n\n“The anticompetitive effects of Google’s scheme cause profound harm to competition, to
consumers, to labor, and to a democratic free press,” reads an announcement posted to the website of the law firm handling the case, Hausfeld.\n\n“Plaintiff
Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure competition for
digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,” it states.\n\nA
Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.']
related_passages: [{'article': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today', 'passage': 'What to know about
Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nTo make its case, Epic plans to bring claims
under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the Cartwright Act, and the Unfair Competition Law.
It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for
payment processing solutions for content inside Android apps.\n\nGoogle argues its commissions aren’t just tied to billing, but offers discount for third-party
billing options\n\nToday, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it
processes.\n\nHowever, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing,
which Epic has declined to use.', 'published_on': '2023-11-06T16:47:11+00:00'}, {'article': 'What to know about Fortnite maker Epic Games’ antitrust battle with
Google, starting today', 'passage': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch,
2023-11-06T16:47:11+00:00)\nThis program, still in pilot testing, is open to all developers who sell apps in the 35 markets where it’s now available, including
the U.S., and reduces the standard commission by 4% for companies who choose to use their own payment processing solution. Spotify and Bumble were the initial
testers for the new system, first introduced in November 2022, and Match, as part of its settlement agreement, will also now take advantage of this
option.\n\nGoogle will also argue that its commissions aren’t just tied to payment processing, but rather help it to develop other features and controls to keep
Android users safe and guide them to discovering new applications, as well as help fund the tools and services developers need to build and grow their apps.
Google also helps developers reach a broader audience beyond the mobile phone, with support for Android across platforms, ranging from tablets to TVs to autos
and more. And it will note that 99% of its developers qualify for a commission of 15% or less, based on their revenues.', 'published_on':
'2023-11-06T16:47:11+00:00'}, {'article': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today', 'passage': 'What to know
about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nThe competitive landscape with other app
stores, OEMs\n\nThe tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out
that not only does the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports
competitive app stores like Samsung’s Galaxy Store and Amazon’s Appstore.\n\nEpic, however, will attempt to demonstrate that Google makes it hard for
alternative app stores to succeed and reach consumers, noting that 90% of apps are still downloaded through Google Play. And it will point out that Google
bundles the Play Store with other Google apps that Android OEMs (original equipment manufacturers) have to agree to in order to use Android.\n\nEpic will also
bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps outside of Google
Play.', 'published_on': '2023-11-06T16:47:11+00:00'}, {'article': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today',
'passage': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nEpic Games will
specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app. But OnePlus had
to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between Google and
Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed by
Samsung.\n\nEpic will argue Google’s developer agreements are anti-competitive\n\nA second set of agreements Epic will highlight are those between Google and
Android app developers themselves. One such program, the “Project Hug” initiative, a part of the Google Games Velocity Program, will be cited as an example of
where Google paid competitors not to distribute their apps on Android outside of the Play Store by targeting those that were most at risk of attrition from
Play, then approaching them with a monetary offer.', 'published_on': '2023-11-06T16:47:11+00:00'}, {'article': 'What to know about Fortnite maker Epic Games’
antitrust battle with Google, starting today', 'passage': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today
(TechCrunch, 2023-11-06T16:47:11+00:00)\nAs an example, Epic will point to Activision Blizzard, the owner of King.com and maker of Candy Crush and Call of Duty,
as one developer that wanted to go the route of opening its own app store. Google and Activision Blizzard then entered into a three-year agreement in January
2020, worth approximately $360 million, to keep the developer on Google Play, Epic will argue.\n\nGoogle, however, will characterize this program as a way to
incentivize developers to launch their apps on Android at the same time as they launch on iOS and plans to dispute Epic’s claims that it prevented developers
from opening their own app stores.\n\nAndroid allows sideloading\n\nGoogle will make note that, unlike Apple, it also allows apps to be sideloaded on Android
devices — something Epic takes advantage of today. It will note this process is streamlined and helps keep users safe.', 'published_on':
'2023-11-06T16:47:11+00:00'}, {'article': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today', 'passage': 'What to know
about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nIt will note this process is streamlined
and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with multiple steps involved and “scare screens”
that warn users of the security risks involved with sideloading.\n\nEpic will also suggest that Google hid items from discovery by allowing execs and employees
to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of
documents.\n\nThe tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the
Play Store’s 2.5 billion users without having to pay.\n\n“Android’s choice and flexibility work well for consumers and developers of all sizes.',
'published_on': '2023-11-06T16:47:11+00:00'}, {'article': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today',
'passage': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\n“Android’s
choice and flexibility work well for consumers and developers of all sizes. We look forward to making our case in court as we fight to keep our users safe from
harm, partner with developers to grow their businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by Wilson
White, VP, Government Affairs & Public Policy.\n\nEpic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement
with the latter last month and with Match last week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in
California’s Northern District, will include testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney,
as well as Apple, Netflix and other Android developers.\n\nGoogle is also involved in another antitrust lawsuit with the Department of Justice over its alleged
search monopoly.', 'published_on': '2023-11-06T16:47:11+00:00'}, {'article': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google,
starting today', 'passage': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch,
2023-11-06T16:47:11+00:00)\nEpic, meanwhile, lost its antitrust battle with Apple and is now asking the Supreme Court to weigh in.', 'published_on':
'2023-11-06T16:47:11+00:00'}, {'article': '5 things we learned so far about the Google antitrust case', 'passage': '5 things we learned so far about the Google
antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nGoogle CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company.
The executive defended Google’s business tactics, including its deal with Apple and other partners to make Google the default search engine.\n\nThe lawsuit
stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the Department of
Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s monopolistic
grip on the market and restore competition to digital advertising.”\n\nThe government has argued that Google uses its platforms and deals with partners to block
out any competition in search or advertising, thus hindering competitors from accessing the data they’d need to improve their products.\n\nGoogle argues that
it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the company says.',
'published_on': '2023-10-31T02:30:32+00:00'}, {'article': '5 things we learned so far about the Google antitrust case', 'passage': '5 things we learned so far
about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nThat doesn’t make its actions illegal, the company says.\n\nMonday’s trial hearing
revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in 2021, how Google tried to
take it further and have Chrome preinstalled on iPhones and more.\n\nGoogle paid $26 billion in 2021 to be everyone’s default search engine\n\nWhen Google’s
search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple browsers, phones and
platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.\n\nAbout $18 billion of that total amount went directly to Apple, according
to a New York Times report published earlier this month. Google has had its deal with Apple in place since 2003.\n\nWhen questioned on the amount of money
Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.', 'published_on':
'2023-10-31T02:30:32+00:00'}, {'article': '5 things we learned so far about the Google antitrust case', 'passage': '5 things we learned so far about the Google
antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nGoogle pays big bucks to be everywhere so that it can take in all the data and be the best search engine
across different companies’ devices, said Pichai, according to The Verge.\n\nGoogle understood the value of defaults very early on. U.S. Justice Department
lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy meeting containing data showing that when people changed their browser homepage
to Google, they did 15% more Google searches. When they switched away, their Google searches dropped 27%.\n\n“Nitin argues that focusing on homepage market
share is one of the most effective things we can do to make gains in search market share,” read an email that summarized the meeting and was sent to Pichai, as
well as other Google leaders, according to The Verge.\n\nThe amount that Google spent on homepage market share has been a fixing point in the trial.',
'published_on': '2023-10-31T02:30:32+00:00'}, {'article': '5 things we learned so far about the Google antitrust case', 'passage': '5 things we learned so far
about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nThe amount that Google spent on homepage market share has been a fixing point in the
trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified that their search engines would have been far more successful, even competitive with
Google, had they been able to make similar deals with Apple. Microsoft CEO Satya Nadella even said he was willing to spend $15 billion per year to get Bing into
Apple’s default search, per The Information.\n\nGoogle agreed not to promote Chrome to Safari users\n\nAs part of its search deal with Apple, Google agreed not
to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with banners, pop-ups and other annoying means in other Google
apps.\n\nThe agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their
iPhones.\n\nGoogle tried to be preinstalled on iPhones\n\nPichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018,
according to The Verge.', 'published_on': '2023-10-31T02:30:32+00:00'}, {'article': '5 things we learned so far about the Google antitrust case', 'passage': '5
things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nHe hoped to make Google and Apple’s services so connected as
to be inseparable.\n\nThe way Pichai pitched it would have been a win-win for both companies. Google gets more people searching on its platform — not to mention
all that juicy data — and Apple would get more revenue, as a result of the lucrative search agreement the two signed.\n\nFor whatever reason, Cook didn’t take
the bait. Apple doesn’t preload third-party software onto its devices, and it wasn’t going to make an exception for Google.\n\nGoogle’s deleted chat
logs\n\nDuring Pichai’s testimony, the DOJ touched on Google’s policy of deleting internal chat messages, despite being subject to a litigation hold. In
February, the DOJ accused Google of systematically destroying chats through its history-off option, which deletes messages every 24 hours unless a user manually
changed the setting.\n\nPichai said that he took action against the history-off default for chat in February to comply with the DOJ’s litigation hold, according
to CNBC.', 'published_on': '2023-10-31T02:30:32+00:00'}, {'article': '5 things we learned so far about the Google antitrust case', 'passage': '5 things we
learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nBellshaw pulled up a message exchange in 2021 where Pichai asked for
history to be turned off in a group chat. Pichai responded that he wanted to discuss a personnel matter and the subject had nothing to do with the litigation
hold, which he said he takes great care to comply with.\n\nA moment of nostalgia for Internet Explorer\n\nCast your memories back to 2005, when Microsoft’s
Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent Microsoft an angry letter, saying that making Internet
Explorer the search default was anticompetitive. Oh, how the tables have turned.\n\nAfter establishing that Google understands the inherent value of defaults,
Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The letter declared that problems with a default setting are made worse by how
changes to defaults are handled, and that most end users “do not change defaults.”', 'published_on': '2023-10-31T02:30:32+00:00'}, {'article': '5 things we
learned so far about the Google antitrust case', 'passage': '5 things we learned so far about the Google antitrust case (TechCrunch,
2023-10-31T02:30:32+00:00)\nThese are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they
accuse Google of being anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced
years ago.\n\nWhat does it all mean?\n\nThe case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since
the U.S. took Microsoft to trial in the 1990s.\n\nIf the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer
company was required to change its behavior and share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive
deals with computer manufacturers.\n\nGoogle might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday.', 'published_on':
'2023-10-31T02:30:32+00:00'}, {'article': '5 things we learned so far about the Google antitrust case', 'passage': '5 things we learned so far about the Google
antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)\nThe DOJ has said that Google gets 16 times more data than Bing does everyday.\n\nThe Google outcome
could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to illegally maintain
its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether it unfairly favors
its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.\n\nEnforcers will
want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.', 'published_on':
'2023-10-31T02:30:32+00:00'}]
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Match may have settled its antitrust lawsuit with Google last week, but Fortnite maker Epic Games is still set to go to trial with the tech giant today,
November 6, in hopes of convincing a jury that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission
structure. This case will differ slightly from Epic’s battle with Apple on the same topic because, this time, Epic can’t allege there’s no other way to load
apps onto Android devices, as it could with Apple, because Android does allow for sideloading apps. Though Epic will suggest Google makes that process
cumbersome, the bulk of its arguments will center around what it believes are anticompetitive agreements between Google and device manufacturers and developers.

The precedent Epic’s case will rely on is a Microsoft case where courts found Microsoft to have a monopoly over the operating system and were abusing it, making
it hard for users to download alternative browsers, like Netscape, on Windows devices.
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=8655a53bf143f722 ===

=== ITEM id=f5caebd6313d39f6 ===
Gold sentences and facts:
Sentence 0: If the debt continues to rise more than the economy, risks will increase, and the government may “crowd out” private investment as it competes with the private sector for funds to service its debt.
  Fact 0.0: If the debt continues to rise more than the economy, risks will increase.
  Fact 0.1: If the debt continues to rise more than the economy, the government may 'crowd out' private investment.
  Fact 0.2: The government may crowd out private investment as it competes with the private sector for funds to service its debt.
Sentence 1: And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63% and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr. told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).
  Fact 1.0: President Ferdinand Marcos, Jr. said 'We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63%'.
  Fact 1.1: President Ferdinand Marcos, Jr. said the Philippines' debt-to-GDP ratio of about 63% is a little high for the country and is not ideal.
  Fact 1.2: President Ferdinand Marcos, Jr. said this to members of the US-ASEAN Business Council who visited Malacañang.
  Fact 1.3: ABS-CBN reported this on Aug. 9, 2023.

Retrieved items:
<item 1>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
the government borrows money either from foreign sources or from the domestic capital market which increases the debt stock of the NG and its debt servicing
requirements; b.) the government borrows money from the Bangko Sentral ng Pilipinas; or, c.) the government withdraws funds from its cash balances in the
Treasury. Historically, National Government expenditures have always exceeded total revenues resulting in annual budget deficits, the DBM declared. “Thus, the
National Government had to resort to borrowing to cover said deficits which resulted in the ballooning of foreign and domestic debts,” the budget office
admitted.

“However, in 1994, the government broke the deficit trend by posting a budget surplus of P16 billion through an aggressive privatization and revenue generation
program and a prudent expenditure program. Since then, the government has been exerting efforts to maintain the surplus budget policy. The surplus budget policy
is important to encourage economic growth,” the DBM said.

Graph facts:
- Strategizing for sustainable development -PUBLISHED_BY-> Business World
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> House of Representatives
- A deficit budget for 2024 -MENTIONS-> deficit
- A deficit budget for 2024 -MENTIONS-> Congress

Entities: Business World, A deficit budget for 2024, Bangko Sentral ng Pilipinas, DBM, budget deficits, foreign and domestic debts, budget surplus policy,
economic growth, privatization, revenue generation, expenditure program, budget surplus of P16 billion, Treasury, National Government
</item 1>

<item 2>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
For this, the economy must grow its per capita income above the threshold set by the World Bank, which means a gross national income (GNI) per capita of at
least $4,466. In 2022, the Philippines achieved a GNI per capita of $3,950.

The second is to lower the poverty level from 18.1% in 2021 to single digits by 2028 — the end of the term of the current administration. Both indicators
require high growth rates. For the next year through 2028, the government pencils the growth rate between 6.5% and 8%.

Growth is expected to be investment-led with the implementation of structural changes such as the Corporate Recovery and Tax Incentives for Enterprises (CREATE)
law, which lowered the corporate income tax rate, and the amendments to the Public Service Act (PSA), Foreign Investment Act (FIA), and Retail Trade
Liberalization Act (RTLA), which further liberalized the economy. The new legislation is expected to attract more local and foreign investment, especially in
the liberalized sectors.

Graph facts:
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -MENTIONS-> World Bank
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> World Bank
- annual meetings of the International Monetary Fund and the World Bank -MENTIONS-> World Bank
- Peace Studies scholar -WORKS_FOR-> World Bank
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -MENTIONS-> poverty

Entities: World Bank, poverty, economy, Business World, gross national income, Corporate Recovery and Tax Incentives for Enterprises law, Public Service Act,
Foreign Investment Act, Retail Trade Liberalization Act, Strategizing for sustainable development, Philippines, investment
</item 2>

<item 3>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
To this end, the first goal of the PDP is social transformation which includes, among others, sustained expenditure on social services, mainly education,
health, and social protection.

The first two goals of the PDP help attain the developmental objectives of efficiency and equity, which rest on the pillars of sustainability and resilience.
The third fosters societal resilience: an enabling environment encompassing institutions, macroeconomic stability, and the physical and natural environment.

Institutions are vital to economic acceleration, which is why the government’s steps to enhance the ease of doing business are most welcome. Infrastructure
development also enables an economy to sustain higher levels of growth, which, in turn, catalyzes yet more investment. The Public-Private Partnership (PPP) Act
is up for the signature of President Marcos and, if approved, is expected to further enhance the business atmosphere in mobilizing private resources for
infrastructure development.

The government aims to sustain its infrastructure program at 5-6% of GDP through 2028 amid a six-year medium-term fiscal framework.

Graph facts:
- There are no sacred numbers -MENTIONS-> GDP
- Israel’s startup ecosystem: down but not out -MENTIONS-> GDP
- India to be one of world's fastest growing economies over next decade -MENTIONS-> GDP
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> education
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -MENTIONS-> education

Entities: GDP, education, economy, Business World, Strategizing for sustainable development, PDP, social transformation, social services, social protection,
equity, institutions, macroeconomic stability, physical and natural environment, ease of doing business, infrastructure development
</item 3>

<item 4>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
The National Government plans to raise more revenue to finance the country’s socio-economic needs, largely through a progressive and simplified tax system, more
efficient and effective tax collection measures through digitalization and, to some extent, from policy measures such as value-added tax on digital service
providers and excise tax on sweetened beverages and junk food.

Achieving fiscal stability presupposes the sustainability of fiscal policy, and fiscal stability helps “crowd in” private investment.

PRICE STABILITY AND INFLATION

In his second State of the Nation Address (SONA), President Ferdinand Marcos, Jr. singled out inflation as the country’s most pressing problem. In the first
nine months of 2023, inflation averaged 6.6% — far above the upper end of the target range of 2-4% set by the BSP. If left unchecked, inflation could undermine
growth. Not only does this increase costs to organizations, but it also sets in motion second-round effects as workers start demanding higher wages,
consequently increasing business costs and discouraging investments.

Graph facts:
- A deficit budget for 2024 -MENTIONS-> BSP
- A deficit budget for 2024 -PUBLISHED_BY-> Business World
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> wages
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> investments

Entities: wages, Business World, BSP, investments, Strategizing for sustainable development, National Government, Ferdinand Marcos, Jr., State of the Nation
Address, fiscal stability, private investment, progressive and simplified tax system, digitalization, value-added tax, digital service providers, excise tax
</item 4>

<item 5>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
But traditional politicians (called “trapo” in the Philippines) might have it in their hearts to think about the country and the welfare of future generations.

How many generations will pay for the monstrous debt incurred for the dubious objectives and vanities of some today?

Amelia H. C. Ylagan is a doctor of Business Administration from the University of the Philippines.

ahcylagan@yahoo.com

Graph facts:
- Strategizing for sustainable development -PUBLISHED_BY-> Business World
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> House of Representatives
- A deficit budget for 2024 -MENTIONS-> deficit
- A deficit budget for 2024 -MENTIONS-> Congress

Entities: Business World, A deficit budget for 2024, deficit budget, Amelia H. C. Ylagan, University of the Philippines, traditional politicians, Business
Administration, Philippines, debt
</item 5>

<item 6>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
During the Congressional budget hearing, Finance Secretary Benjamin Diokno optimistically provided updates on the government’s fiscal and revenue collection
performances and presented sources of funding for the national budget. Tax revenue is projected to increase from P3.5 trillion in 2023 to P6.5 trillion in 2028,
while non-tax revenues are expected to decline from P191 billion in 2023 to P184 billion in 2028, Diokno told the Philippine News Agency (pna.gov.ph, Aug. 10).

Diokno said the Finance department aims to gradually reduce net financing or new debt from P1.42 trillion in 2022 to P1.22 trillion in 2024, which will help
bring the country’s debt-to-GDP ratio to almost below 60% by 2025. And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63%
and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr.

Graph facts:
- Strategizing for sustainable development -PUBLISHED_BY-> Business World
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> House of Representatives
- A deficit budget for 2024 -MENTIONS-> deficit
- A deficit budget for 2024 -MENTIONS-> Congress

Entities: Business World, A deficit budget for 2024, Benjamin Diokno, Congressional budget hearing, Finance department, Philippine News Agency, national budget,
debt-to-GDP ratio, Ferdinand Marcos, Jr., government, Philippines
</item 6>

<item 7>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
This plan gradually narrows the deficit to 3% of GDP by 2028, down from 7.3% in 2022, such that the debt-to-GDP ratio is reduced from nearly 61% in 2022 to a
more sustainable level of at most 53% by 2028.

DEBT MANAGEMENT

The National Government’s debt was less than 40% before the pandemic struck. It expended much of its fiscal space combatting the pandemic, incurring debt and
large budget deficits. While the current 61% debt-to-GDP ratio may be manageable for an emerging economy like the Philippines for some time, the country may not
have the fiscal space to respond to another potential domestic or external shock. If the debt continues to rise more than the economy, risks will increase, and
the government may “crowd out” private investment as it competes with the private sector for funds to service its debt.

The planned fiscal consolidation entails harmonizing the revenue needs with the promotion of investment through the structure and administration of the reformed
incentive system.

Graph facts:
- A deficit budget for 2024 -MENTIONS-> debt-to-GDP ratio
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> budget deficits
- A deficit budget for 2024 -PUBLISHED_BY-> Business World
- Ferdinand Marcos, Jr. -SAID-> debt-to-GDP ratio

Entities: Business World, budget deficits, debt-to-GDP ratio, Strategizing for sustainable development, National Government, private sector, fiscal space,
fiscal consolidation, reformed incentive system, Philippines, investment, pandemic
</item 7>

<item 8>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).

Whom are we to believe, but inflation — which independently says it as it is.

“While inflation in the Philippines has been on a downward trend since January, the country needs to achieve a 3.7% average inflation rate for the next five
months for the country to hit the BSP’s forecast of 5.5% inflation for 2023, Assistant National Statistician Rachel Lacsa said (Ibid.). Some economists say that
deficit budgets and the remedies — monetary (borrowings increasing money supply) and fiscal (like raising taxes) increase inflation, and stubborn inflation
further entrenches structural deficits in the system. It is a vicious cycle.”

National budgets might not be innocent of political agendas (call them plans and programs) that can divert money from the proper socio-economic goals of
equilibrium as a balanced budget might work towards.

Graph facts:
- Strategizing for sustainable development -PUBLISHED_BY-> Business World
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> House of Representatives
- A deficit budget for 2024 -MENTIONS-> deficit
- A deficit budget for 2024 -MENTIONS-> Congress

Entities: Business World, A deficit budget for 2024, balanced budget, ABS-CBN, US-ASEAN Business Council, Malacañang, Rachel Lacsa, BSP, deficit budgets,
monetary remedies, borrowings, money supply, fiscal remedies, raising taxes, structural deficits
</item 8>

<item 9>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
Inflation is partly driven by supply-side issues and the government is allocating more resources to the agriculture sector to boost production. Mr. Marcos also
mentioned that the National Government had distributed 28,000 new tools and machinery to farmers. An additional 600 km of farm-to-market roads were laid down to
support the 14 million hectares of farmland, enhancing farmer access to markets. In addition, he signed Executive Order No. 28 in May, forming the Inter-Agency
Committee on Inflation and Market Outlook, which is tasked to keep inflation within government targets and boost the economy.

ENHANCING CLIMATE RESILIENCE

While price stability and fiscal sustainability are important macroeconomic issues, environmental sustainability is increasingly gaining importance. Climate
change uncertainties and challenges need to be managed and both the Philippine government and the private sector are hard-pressed to deliver their commitments
to addressing them. Despite climate change risks, the Philippines has the opportunity to position itself as a prime destination of foreign investments against
climate change or environmental, social, and governance (ESG) investments.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> economy
- ASX to open lower; Wall Street falls as the clamp tightens from rising bond yields -MENTIONS-> economy
- ASX set to open lower as Wall Street closes September with more losses -MENTIONS-> economy
- There are no sacred numbers -MENTIONS-> economy
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> economy

Entities: economy, Business World, Strategizing for sustainable development, Mr. Marcos, Philippine government, supply-side issues, agriculture sector, Signing
Executive Order No. 28, Inter-Agency Committee on Inflation and Market Outlook, fiscal sustainability, environmental sustainability, private sector, foreign
investments, environmental, social, and governance (ESG) investments, Philippines
</item 9>

<item 10>
A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
Why then, do we have a deficit budget for 2024?

The national budget for 2024 is at P5.768 trillion, 9.5% higher than the current (2023) budget of P5.268 trillion. Observers note that this first budget fully
prepared by the current administration accounts for 21.7% of gross domestic product (GDP). The 2024 expenditure program is higher by P499.6 billion or a
year-on-year growth rate of 9.5% as compared to the 2023 level of P5.268 trillion. The P5.768-trillion budget, to be supported mainly by P3.5-trillion tax
revenues (based on 2023 tax revenue projections), would mean a budget deficit of some P2.268 trillion, to be funded by additional borrowings. This could
increase government debt from the current P14.24 trillion (as of July) to P16+ trillion by the end of 2024.

Graph facts:
- Strategizing for sustainable development -PUBLISHED_BY-> Business World
- From FAPE/PEAC to PhilEd -PUBLISHED_BY-> Business World
- A deficit budget for 2024 -MENTIONS-> House of Representatives
- A deficit budget for 2024 -MENTIONS-> deficit
- A deficit budget for 2024 -MENTIONS-> Congress

Entities: Business World, A deficit budget for 2024, budget deficit, national budget for 2024, 2023 budget, current administration, 2024 expenditure program,
tax revenues, additional borrowings, government debt, gross domestic product
</item 10>
=== END ITEM id=f5caebd6313d39f6 ===
