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

=== ITEM id=096e710ce226b884 ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 0.1: The case argues that Google “siphons off” news publishers’ content.
  Fact 0.2: The case argues that Google siphons off news publishers’ readers.
  Fact 0.3: The case argues that Google siphons off news publishers’ ad revenue.
  Fact 0.4: The case argues that Google does this through anticompetitive means.
Sentence 1: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 1.0: Epic will also suggest that Google hid items from discovery.
  Fact 1.1: Epic will suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats.
  Fact 1.2: Google will counter this as well.
  Fact 1.3: Google will say it has provided the court with thousands of chat logs.
  Fact 1.4: Google will say it has provided the court with millions of documents.

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
</item 3>

<item 4>
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
</item 4>

<item 5>
Graph query row:
TechCrunch_report: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
TechCrunch_passages: ['News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch,
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
TechCrunch_related_entities: [{'entity': 'Alphabet', 'relationship': 'MENTIONS'}, {'entity': 'Helena World Chronicle', 'relationship': 'MENTIONS'}, {'entity':
'U.S. District Court in D.C.', 'relationship': 'MENTIONS'}, {'entity': 'Search Generative Experience', 'relationship': 'MENTIONS'}, {'entity': 'Bard AI
chatbot', 'relationship': 'MENTIONS'}, {'entity': 'Knowledge Graph', 'relationship': 'MENTIONS'}, {'entity': 'anticompetitive behavior', 'relationship':
'MENTIONS'}, {'entity': 'U.S. antitrust law', 'relationship': 'MENTIONS'}, {'entity': 'Sherman Act', 'relationship': 'MENTIONS'}, {'entity': 'Arkansas',
'relationship': 'MENTIONS'}, {'entity': 'question-and-answer technologies', 'relationship': 'MENTIONS'}, {'entity': 'Publishers', 'relationship': 'MENTIONS'},
{'entity': 'Knowledge Panel', 'relationship': 'MENTIONS'}, {'entity': 'Featured Snippets', 'relationship': 'MENTIONS'}, {'entity': 'class action antitrust
suit', 'relationship': 'MENTIONS'}, {'entity': 'misappropriation of Publishers’ content', 'relationship': 'MENTIONS'}, {'entity': 'The Wall Street Journal',
'relationship': 'MENTIONS'}, {'entity': 'report on Thursday', 'relationship': 'MENTIONS'}, {'entity': 'deal to license news for AI model training',
'relationship': 'MENTIONS'}, {'entity': 'SGE', 'relationship': 'MENTIONS'}, {'entity': 'Google Bard AI', 'relationship': 'MENTIONS'}, {'entity': 'GoogleBot',
'relationship': 'MENTIONS'}, {'entity': 'Google general search service', 'relationship': 'MENTIONS'}, {'entity': 'AdSense', 'relationship': 'MENTIONS'},
{'entity': 'News Media Alliance', 'relationship': 'MENTIONS'}, {'entity': 'Washington Post', 'relationship': 'MENTIONS'}, {'entity': 'Epic Games lawsuit against
Google over app store antitrust issues', 'relationship': 'MENTIONS'}, {'entity': 'app store antitrust issues', 'relationship': 'MENTIONS'}, {'entity': 'U.S.
Justice Department', 'relationship': 'MENTIONS'}, {'entity': 'Hausfeld', 'relationship': 'MENTIONS'}, {'entity': 'Helena World Chronicle, LLC', 'relationship':
'MENTIONS'}, {'entity': 'class action antitrust suit against Google', 'relationship': 'MENTIONS'}, {'entity': 'U.S. Justice Department lawsuit against Google',
'relationship': 'MENTIONS'}, {'entity': '2020 Justice Department civil antitrust suit over search and search advertising', 'relationship': 'MENTIONS'},
{'entity': 'digital ad technologies', 'relationship': 'MENTIONS'}, {'entity': 'Clayton Act', 'relationship': 'MENTIONS'}, {'entity': 'search and search
advertising', 'relationship': 'MENTIONS'}, {'entity': 'digital news and reference publishing', 'relationship': 'MENTIONS'}, {'entity': 'class-wide monetary and
injunctive relief', 'relationship': 'MENTIONS'}, {'entity': 'free marketplace of ideas', 'relationship': 'MENTIONS'}, {'entity': 'free press', 'relationship':
'MENTIONS'}, {'entity': 'Epic Games', 'relationship': 'MENTIONS'}, {'entity': 'antitrust', 'relationship': 'MENTIONS'}, {'entity': 'The Atlantic',
'relationship': 'MENTIONS'}, {'entity': 'Axel Springer', 'relationship': 'MENTIONS'}, {'entity': 'Allen Institute for AI', 'relationship': 'MENTIONS'},
{'entity': 'class action lawsuit', 'relationship': 'MENTIONS'}, {'entity': 'consumers', 'relationship': 'MENTIONS'}, {'entity': 'AI', 'relationship':
'MENTIONS'}, {'entity': 'AI technologies', 'relationship': 'MENTIONS'}, {'entity': 'Similarweb', 'relationship': 'MENTIONS'}, {'entity': 'Google',
'relationship': 'MENTIONS'}, {'entity': 'artificial intelligence', 'relationship': 'MENTIONS'}, {'entity': 'D.C.', 'relationship': 'MENTIONS'}, {'entity':
'OpenAI', 'relationship': 'MENTIONS'}, {'entity': 'labor', 'relationship': 'MENTIONS'}]
publishers: ['TechCrunch']
Epic_Games_report: What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today
Epic_Games_passages: ['What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nTo
make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.\n\nGoogle argues its commissions aren’t
just tied to billing, but offers discount for third-party billing options\n\nToday, Google requires apps to use its own first-party billing system, and charges
app developers a 15% to 30% commission on the sales it processes.\n\nHowever, Google will counter Epic’s arguments by pointing out that it has rolled out a new
option for app developers called User Choice Billing, which Epic has declined to use.', 'What to know about Fortnite maker Epic Games’ antitrust battle with
Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nThis program, still in pilot testing, is open to all developers who sell apps in the 35 markets
where it’s now available, including the U.S., and reduces the standard commission by 4% for companies who choose to use their own payment processing solution.
Spotify and Bumble were the initial testers for the new system, first introduced in November 2022, and Match, as part of its settlement agreement, will also now
take advantage of this option.\n\nGoogle will also argue that its commissions aren’t just tied to payment processing, but rather help it to develop other
features and controls to keep Android users safe and guide them to discovering new applications, as well as help fund the tools and services developers need to
build and grow their apps. Google also helps developers reach a broader audience beyond the mobile phone, with support for Android across platforms, ranging
from tablets to TVs to autos and more. And it will note that 99% of its developers qualify for a commission of 15% or less, based on their revenues.', 'What to
know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nThe competitive landscape with other
app stores, OEMs\n\nThe tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point
out that not only does the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also
supports competitive app stores like Samsung’s Galaxy Store and Amazon’s Appstore.\n\nEpic, however, will attempt to demonstrate that Google makes it hard for
alternative app stores to succeed and reach consumers, noting that 90% of apps are still downloaded through Google Play. And it will point out that Google
bundles the Play Store with other Google apps that Android OEMs (original equipment manufacturers) have to agree to in order to use Android.\n\nEpic will also
bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps outside of Google
Play.', 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nEpic Games will
specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app. But OnePlus had
to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between Google and
Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed by
Samsung.\n\nEpic will argue Google’s developer agreements are anti-competitive\n\nA second set of agreements Epic will highlight are those between Google and
Android app developers themselves. One such program, the “Project Hug” initiative, a part of the Google Games Velocity Program, will be cited as an example of
where Google paid competitors not to distribute their apps on Android outside of the Play Store by targeting those that were most at risk of attrition from
Play, then approaching them with a monetary offer.', 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch,
2023-11-06T16:47:11+00:00)\nAs an example, Epic will point to Activision Blizzard, the owner of King.com and maker of Candy Crush and Call of Duty, as one
developer that wanted to go the route of opening its own app store. Google and Activision Blizzard then entered into a three-year agreement in January 2020,
worth approximately $360 million, to keep the developer on Google Play, Epic will argue.\n\nGoogle, however, will characterize this program as a way to
incentivize developers to launch their apps on Android at the same time as they launch on iOS and plans to dispute Epic’s claims that it prevented developers
from opening their own app stores.\n\nAndroid allows sideloading\n\nGoogle will make note that, unlike Apple, it also allows apps to be sideloaded on Android
devices — something Epic takes advantage of today. It will note this process is streamlined and helps keep users safe.', 'What to know about Fortnite maker Epic
Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nIt will note this process is streamlined and helps keep users safe.
But Epic will claim that this process is made to be overly difficult for end users, with multiple steps involved and “scare screens” that warn users of the
security risks involved with sideloading.\n\nEpic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the
record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.\n\nThe tech giant
will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5 billion users
without having to pay.\n\n“Android’s choice and flexibility work well for consumers and developers of all sizes.', 'What to know about Fortnite maker Epic
Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\n“Android’s choice and flexibility work well for consumers and
developers of all sizes. We look forward to making our case in court as we fight to keep our users safe from harm, partner with developers to grow their
businesses, and keep the Android ecosystem thriving and healthy for everyone,” reads a Google blog post by Wilson White, VP, Government Affairs & Public
Policy.\n\nEpic’s lawsuit originally involved Match and several U.S. attorneys general, but Google reached a settlement with the latter last month and with
Match last week. Epic will now fight the antitrust battle on its own. The lawsuit, held in the U.S. District Court in California’s Northern District, will
include testimony from a number of Google and Android execs, including CEO Sundar Pichai, and Epic Games CEO Tim Sweeney, as well as Apple, Netflix and other
Android developers.\n\nGoogle is also involved in another antitrust lawsuit with the Department of Justice over its alleged search monopoly.', 'What to know
about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)\nEpic, meanwhile, lost its antitrust
battle with Apple and is now asking the Supreme Court to weigh in.']
Epic_Games_related_entities: [{'entity': 'Sundar Pichai', 'relationship': 'MENTIONS'}, {'entity': 'U.S. antitrust law', 'relationship': 'MENTIONS'}, {'entity':
'Sherman Act', 'relationship': 'MENTIONS'}, {'entity': 'User Choice Billing', 'relationship': 'MENTIONS'}, {'entity': 'Google first-party billing system',
'relationship': 'MENTIONS'}, {'entity': 'Epic Games’ antitrust battle with Google', 'relationship': 'MENTIONS'}, {'entity': 'California’s anticompetitive laws',
'relationship': 'MENTIONS'}, {'entity': 'Cartwright Act', 'relationship': 'MENTIONS'}, {'entity': 'Unfair Competition Law', 'relationship': 'MENTIONS'},
{'entity': 'distribution of apps to Android users', 'relationship': 'MENTIONS'}, {'entity': 'payment processing solutions for content inside Android apps',
'relationship': 'MENTIONS'}, {'entity': 'Google Play', 'relationship': 'MENTIONS'}, {'entity': 'App Store', 'relationship': 'MENTIONS'}, {'entity': 'Android',
'relationship': 'MENTIONS'}, {'entity': 'Galaxy Store', 'relationship': 'MENTIONS'}, {'entity': 'Amazon Appstore', 'relationship': 'MENTIONS'}, {'entity':
'Android OEMs', 'relationship': 'MENTIONS'}, {'entity': 'Google apps', 'relationship': 'MENTIONS'}, {'entity': 'Anti-Fragmentation Agreements', 'relationship':
'MENTIONS'}, {'entity': 'Ninth Circuit', 'relationship': 'MENTIONS'}, {'entity': 'Ninth Circuit ruling in the Apple case', 'relationship': 'MENTIONS'},
{'entity': 'App store competition', 'relationship': 'MENTIONS'}, {'entity': 'OnePlus', 'relationship': 'MENTIONS'}, {'entity': 'Epic Games app', 'relationship':
'MENTIONS'}, {'entity': 'Google Play Store', 'relationship': 'MENTIONS'}, {'entity': 'Android app developers', 'relationship': 'MENTIONS'}, {'entity': 'Project
Hug', 'relationship': 'MENTIONS'}, {'entity': 'Google Games Velocity Program', 'relationship': 'MENTIONS'}, {'entity': 'antitrust battle', 'relationship':
'MENTIONS'}, {'entity': 'agreement with OnePlus', 'relationship': 'MENTIONS'}, {'entity': 'deal between Google and Samsung', 'relationship': 'MENTIONS'},
{'entity': 'Google developer agreements', 'relationship': 'MENTIONS'}, {'entity': 'anti-competitive practices', 'relationship': 'MENTIONS'}, {'entity':
'Activision Blizzard', 'relationship': 'MENTIONS'}, {'entity': 'King.com', 'relationship': 'MENTIONS'}, {'entity': 'Candy Crush', 'relationship': 'MENTIONS'},
{'entity': 'three-year agreement', 'relationship': 'MENTIONS'}, {'entity': 'Epic Games', 'relationship': 'MENTIONS'}, {'entity': 'Sideloading', 'relationship':
'MENTIONS'}, {'entity': 'Security risks', 'relationship': 'MENTIONS'}, {'entity': 'Off-the-record chats', 'relationship': 'MENTIONS'}, {'entity': 'Chat logs and
documents provided to the court', 'relationship': 'MENTIONS'}, {'entity': 'Play Store access and payment', 'relationship': 'MENTIONS'}, {'entity': 'Wilson
White', 'relationship': 'MENTIONS'}, {'entity': 'Match', 'relationship': 'MENTIONS'}, {'entity': 'U.S. attorneys general', 'relationship': 'MENTIONS'},
{'entity': "Epic's antitrust lawsuit", 'relationship': 'MENTIONS'}, {'entity': 'U.S. District Court in California’s Northern District', 'relationship':
'MENTIONS'}, {'entity': 'Department of Justice', 'relationship': 'MENTIONS'}, {'entity': "Google's antitrust lawsuit with the Department of Justice",
'relationship': 'MENTIONS'}, {'entity': 'search monopoly', 'relationship': 'MENTIONS'}, {'entity': 'antitrust', 'relationship': 'MENTIONS'}, {'entity':
'antitrust battle with Google', 'relationship': 'MENTIONS'}, {'entity': 'antitrust battle with Apple', 'relationship': 'MENTIONS'}, {'entity': 'Call of Duty',
'relationship': 'MENTIONS'}, {'entity': 'Netflix', 'relationship': 'MENTIONS'}, {'entity': 'Apple', 'relationship': 'MENTIONS'}, {'entity': 'Supreme Court',
'relationship': 'MENTIONS'}, {'entity': 'Bumble', 'relationship': 'MENTIONS'}, {'entity': 'Google Play billing', 'relationship': 'MENTIONS'}, {'entity': 'Tim
Sweeney', 'relationship': 'MENTIONS'}, {'entity': 'Samsung', 'relationship': 'MENTIONS'}, {'entity': 'Google', 'relationship': 'MENTIONS'}, {'entity': 'iOS',
'relationship': 'MENTIONS'}, {'entity': 'competition', 'relationship': 'MENTIONS'}, {'entity': 'Play Store', 'relationship': 'MENTIONS'}, {'entity': 'Spotify',
'relationship': 'MENTIONS'}, {'entity': 'U.S.', 'relationship': 'MENTIONS'}, {'entity': 'Amazon', 'relationship': 'MENTIONS'}, {'entity': 'Fortnite',
'relationship': 'MENTIONS'}]
shared_Google_entity: Google
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
</item 7>

<item 8>
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
</item 8>

<item 9>
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
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=096e710ce226b884 ===

=== ITEM id=601862051d45623d ===
Gold sentences and facts:
Sentence 0: McHenry himself has brushed off attempts to take the job more permanently after he was appointed to the role after the unprecedented ouster of McCarthy more than two weeks ago.
  Fact 0.0: McHenry has brushed off attempts to take the job more permanently.
  Fact 0.1: McHenry was appointed to the role after the ouster of McCarthy.
  Fact 0.2: The ouster of McCarthy was unprecedented.
  Fact 0.3: The ouster of McCarthy happened more than two weeks ago.
Sentence 1: Wall Street is also absorbing the ouster of Kevin McCarthy as the speaker of the House of Representatives.
  Fact 1.0: Wall Street is also absorbing the ouster of Kevin McCarthy.
  Fact 1.1: Kevin McCarthy was the speaker of the House of Representatives.
  Fact 1.2: Kevin McCarthy was ousted as speaker of the House of Representatives.

Retrieved items:
<item 1>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
Republicans dropped Rep. Jim Jordan on Friday as their nominee for House speaker, making the decision during a closed-door session after the hard-edged ally of
Donald Trump failed badly on a third ballot for the gavel.

Afterward, Jordan said simply of his colleagues, “We put the question to them, they made a different decision.”

The hard-charging Judiciary Committee chairman said House Republicans now need to come together and “figure out who our speaker is going to be.”

The House impasse deepening into a full-blown crisis, Republicans have no realistic or workable plan to unite the fractured GOP majority, elect a new speaker
and return to the work of Congress that has been languishing since hard-liners ousted Kevin McCarthy at the start of the month.

Majority Leader Steve Scalise said they’re going “come back and start over” on Monday.

Angry, frustrated Republicans who have been watching their majority control descend into chaos left the private session blaming one another for the divisions
they have created.

Graph facts:
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Kevin McCarthy
- Republicans just dumped Jim Jordan again. They're looking for another speaker candidate -MENTIONS-> Kevin McCarthy
- Kevin McCarthy -WORKS_FOR-> House of Representatives
- Kevin McCarthy -PARTICIPATED_IN-> ousting of Kevin McCarthy as the speaker of the House of Representatives
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Kevin McCarthy, Fortune, Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again, Steve Scalise, hard-liners, third
ballot for the gavel, closed-door session, Kevin McCarthy's ousting, come back and start over, House speaker, Congress, Jim Jordan, Judiciary Committee,
Republicans, House
</item 1>

<item 2>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
“We have a chance today to end that chaos, end that uncertainty,” Cole said.

He said that Jordan was not a “shrinking violet” but someone who could lead the House.

Democratic Rep. Pete Aguilar of California nominated Jeffries, noting the Democratic leader continues to win more votes and is the best choice to move the
country forward.

“The country cannot afford more delays and more chaos,” Aguilar said.

Jordan had relied on backing from Trump, the party’s frontrunner in the 2024 election to challenge President Joe Biden, and groups pressuring rank-and-file
lawmakers for the vote, but it was not enough.

Flexing their independence, the holdouts are a mix of pragmatists — ranging from seasoned legislators and committee chairs worried about governing, to newer
lawmakers from districts where voters back home prefer President Joe Biden to Trump.

Some Republicans resent being pressured by Jordan’s allies and say they are being threatened with primary opponents if they don’t support him as speaker. Others
are simply upset at the way the whole process has dragged out.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Joe Biden, California, Trump, Jim Jordan, Republicans, House, Jeffries, Pete Aguilar, 2024 election, Republicans just dumped Jim Jordan
again. They're looking for another speaker candidate, Cole
</item 2>

<item 3>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
Others are simply upset at the way the whole process has dragged out.

They cast their ballots for McCarthy, Majority Leader Steve Scalise — who had been the party’s first nominee to replace McCarthy — and others, one vote even
going to the retired Boehner.

“Jim Jordan will be a great speaker,” Trump had said Tuesday said outside a courthouse in Manhattan, where he is facing business fraud charges. “I think he’s
going to have the votes soon, if not today, over the next day or two.”

Jordan has been a top Trump ally, particularly during the Jan. 6 Capitol attack by the former president’s backers who were trying to overturn the 2020 election
he lost to Biden. Days later, Trump awarded Jordan a Medal of Freedom.

The political climb has been steep for Jordan who is known more as a chaos agent than a skilled legislator, raising questions about how he would lead.

Graph facts:
- Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again -MENTIONS-> 2020 election
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> 2020 election
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune

Entities: 2020 election, Fortune, Steve Scalise, Jan. 6 Capitol attack, Capitol, McCarthy, Biden, Trump, Manhattan, Jim Jordan, Republicans, Republicans just
dumped Jim Jordan again. They're looking for another speaker candidate, Boehner, Medal of Freedom, business fraud
</item 3>

<item 4>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
No further votes were scheduled.

The holdouts added to a surprisingly large and politically diverse group of 20 Republicans who had rejected Jordan’s nomination the day before, many resenting
the hardball tactics seeking to enforce support, and viewing the Ohio congressman as too extreme for a central seat of U.S. power.

With Republicans in majority control of the House, 221-212, Jordan must pick up most of his GOP foes to win. Wednesday’s tally, with 199 Republicans voting for
Jordan and 212 for Democratic leader Hakeem Jeffries of New York, left no candidate with a clear majority, as the 22 Republicans voted for someone else.

One new Jordan opponent, Rep. Vern Buchanan of Florida, explained his vote, “I think it’s time to move on.”

Bipartisan groups of lawmakers have been floating ways to operate the House by giving greater power to McHenry or another temporary speaker. The House had never
ousted its speaker before McCarthy, and McHenry could tap the temporary powers that were created after the Sept.

Graph facts:
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> New York
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> New York
- How the conspiracy-fueled Epoch Times went mainstream and made millions -MENTIONS-> New York
- AfroFuture’s vision is to bring African art, cuisine, fashion, and music to a global audience. -MENTIONS-> New York
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> New York

Entities: New York, Fortune, McCarthy, Jim Jordan, Republicans just dumped Jim Jordan again., Republicans, House, Hakeem Jeffries, Vern Buchanan, McHenry,
Wednesday’s tally, Jordan’s nomination, Ohio, Florida
</item 4>

<item 5>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
A closed-door meeting Thursday to regroup grew heated at times with Republican factions blaming one another for sending their majority into chaos, lawmakers
said.

Elevating McHenry to an expanded speaker’s role could be a possible off-ramp for the crisis, but it would not be as politically simple as it might seem.

Republicans are loath to partner with the Democrats in a bipartisan way on the arrangement, and it’s highly unlikely Republicans could agree to give McHenry
more powers on their own, since their hard-liners don’t like it.

McHenry himself has brushed off attempts to take the job more permanently after he was appointed to the role after the unprecedented ouster of McCarthy more
than two weeks ago.

To win over GOP colleagues, Jordan had relied on backing from Trump, the party’s front-runner in the 2024 election, and groups pressuring rank-and-file
lawmakers for the vote. But they were not enough and in fact backfired on some.

Jordan has been a top Trump ally, particularly during the Jan.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again, Congress, closed-door meeting, ouster of
McCarthy, McCarthy, Trump, Democrats, Jim Jordan, Republicans, House, McHenry, 2024 election
</item 5>

<item 6>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
Once a formality in Congress, the vote for House speaker has devolved into another bitter GOP showdown for the gavel.

As Republicans upset and exhausted by the infighting retreated for private conversations, hundreds of protesters, if not more, amassed outside the Capitol over
the Israel-Hamas war, a stark reminder of the dangers of having the House adrift as political challenges intensify at home and abroad.

Ahead of the morning vote, Jordan, the combative Judiciary Committee chairman, made an unusual plea for party unity — almost daring his colleagues to put
forward the alternative proposal for a temporary speaker.

“We’ve been at this two weeks,” Jordan said at the Capitol. “American people deserve to have their government functioning.”

But as the roll call got under way, he lost more than he gained, picking up three backers but adding more detractors. No further votes were scheduled.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Congress, Capitol, Jim Jordan, Judiciary Committee, Israel-Hamas war, vote for House speaker, Republicans, House, Republicans just dumped Jim
Jordan again. They're looking for another speaker candidate
</item 6>

<item 7>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
Republican Rep. Jim Jordan failed again Wednesday on a crucial second ballot to become House speaker, the hard-fighting ally of Donald Trump losing even more
GOP colleagues who are refusing to give him the gavel.

Next steps were highly uncertain as angry, frustrated Republicans looked at other options. A bipartisan group of lawmakers floated an extraordinary plan — to
give the interim speaker-pro-tempore, Rep. Patrick McHenry, R-N.C., more power to reopen the immobilized House and temporarily conduct routine business.

What was clear was that Jordan’s path to become House speaker was almost certainly lost. He was opposed by 22 Republicans, two more than he lost in first round
voting the day before.

“We’ll keep talking to members, keep working on it,” Jordan, a founding member of the hard-right Freedom Caucus, said after the vote, vowing to stay in the
race.

The House came to another abrupt standstill, stuck now 15 days without a speaker — a position of power second in line to the presidency — since the sudden
ouster of Kevin McCarthy.

Graph facts:
- Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again -MENTIONS-> Kevin McCarthy
- ASX set to open higher as Wall Street rebounds; $A rises -MENTIONS-> Kevin McCarthy
- Kevin McCarthy -WORKS_FOR-> House of Representatives
- Kevin McCarthy -PARTICIPATED_IN-> ousting of Kevin McCarthy as the speaker of the House of Representatives
- Kevin McCarthy -PARTICIPATED_IN-> Kevin McCarthy's ousting

Entities: Kevin McCarthy, Fortune, Jim Jordan, Patrick McHenry, Freedom Caucus, second ballot to become House speaker, House, Republicans just dumped Jim Jordan
again. They're looking for another speaker candidate, Donald Trump
</item 7>

<item 8>
Republicans just dumped Jim Jordan again. They're looking for another speaker candidate (Fortune, 2023-10-18T20:36:54+00:00)
11, 2001, terror attacks to ensure continuity of government.

The novel concept of boosting the interim speaker’s role was gaining favor with a pair of high-profile Republicans: Former GOP speakers Newt Gingrich and John
Boehner.

Gingrich said while he likes Jordan, he has “no faith” the nominee can get much beyond the 200 votes he won in the first vote.

“We can’t sit around and suck our thumbs and hope the world will wait until the House Republicans get their act together,” Gingrich told Fox News’ Sean Hannity
on his show.

Boehner reposted Gingrich’s views saying, “I agree,” on social media.

The two men have deep experience with the subject. Both were chased to early retirement.

“The Republicans are unable to function right now,” Jeffries said late Tuesday. “All options are on the table to end the Republican civil war,” he added
Wednesday.

In nominating Jordan, veteran Republican Rep. Tom Cole of Oklahoma said it was time to end the upheaval that he had warned against with McCarthy’s sudden
ouster.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, McCarthy, Jim Jordan, House Republicans, Newt Gingrich, John Boehner, Fox News, Sean Hannity, Jeffries, Tom Cole, Oklahoma, Jordan
nomination, Republicans just dumped Jim Jordan again. They're looking for another speaker candidate
</item 8>

<item 9>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
Next steps are highly uncertain as lawmakers start offering new ideas for a possible speaker. But it appears no one at present can win a GOP majority.

“We’re in a very bad place right now,” McCarthy said earlier.

In a floor vote Friday morning, Jordan’s third reach for the gavel, he lost 25 Republican colleagues, worse than he had fared earlier in the week, and far from
the majority needed, as they reject his hardline approach.

But after two failed votes, Jordan’s third attempt at the gavel essentially collapsed — in large part because more centrist Republicans are revolting over the
nominee and the hardball tactics being used to win their votes. They have been bombarded with harassing phone calls and even reported death threats.

Ahead of the vote, Jordan showed no signs of stepping aside, insisting at a Capitol press conference: “The American people are hungry for change.”

But afterward Jordan told colleagues behind closed doors, “I tried my best,” according to one of his opponents, Rep. Don Bacon, R-Neb.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again, Don Bacon, Jordan’s third attempt for the gavel,
hardline approach, harassing phone calls, death threats, change, hardball tactics, Congress, Capitol, McCarthy, Jim Jordan, Republicans
</item 9>

<item 10>
Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again (Fortune, 2023-10-20T19:10:22+00:00)
Don Bacon, R-Neb.

For more than two weeks the stalemate has shut down the U.S. House, leaving a seat of American democracy severely hobbled at a time of challenges at home and
abroad. The House Republican majority appears to have no idea how to end the political turmoil and get back to work.

With Republicans in majority control of the House, 221-212, any candidate can lose only a few detractors. It appears there is no Republican at present who can
win a clear majority, 217 votes, to become speaker.

In fact, the hard-charging Judiciary chairman lost rather than gained votes despite hours spent trying to win over holdouts, no improvement from the 20 and then
22 Republicans he lost in early rounds this week.

Friday’s vote was 194 for Jordan, his lowest tally yet, and 210 for Jeffries, with two absences on each side.

“He doesn’t have the votes to be speaker,” Rep.

Graph facts:
- Sen. Robert Menendez, facing bribery charges, stumps bank CEOs by asking how much their firms paid in settlements for allegedly ripping off consumers
-PUBLISHED_BY-> Fortune
- Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style -PUBLISHED_BY-> Fortune
- Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar -PUBLISHED_BY-> Fortune
- Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' -PUBLISHED_BY-> Fortune
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -PUBLISHED_BY-> Fortune

Entities: Fortune, Republicans dump Jim Jordan as House Speaker nominee, leaving Congress in crisis yet again, Don Bacon, U.S. House, Friday’s vote, political
turmoil, Congress, Jim Jordan, Republicans, Jeffries
</item 10>
=== END ITEM id=601862051d45623d ===
