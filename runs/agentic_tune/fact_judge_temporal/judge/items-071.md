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

=== ITEM id=0fb87abb479bfa90 ===
Gold sentences and facts:
Sentence 0: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as well, saying it has provided the court with thousands of chat logs and millions of documents.
  Fact 0.0: Epic will also suggest that Google hid items from discovery by allowing execs and employees to have 'off the record' chats.
  Fact 0.1: Google will counter this as well.
  Fact 0.2: Google will say it has provided the court with thousands of chat logs.
  Fact 0.3: Google will say it has provided the court with millions of documents.
Sentence 1: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 1.0: The full model, Gemini Ultra, was not released.
  Fact 1.1: Only a 'lite' version called Gemini Pro was released.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

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
title: News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line
published_on: 2023-12-15T17:56:02+00:00
publisher: TechCrunch
named_entities: [{'entity': 'Alphabet', 'relationship': 'MENTIONS'}, {'entity': 'Helena World Chronicle', 'relationship': 'MENTIONS'}, {'entity': 'U.S. District
Court in D.C.', 'relationship': 'MENTIONS'}, {'entity': 'Search Generative Experience', 'relationship': 'MENTIONS'}, {'entity': 'Bard AI chatbot',
'relationship': 'MENTIONS'}, {'entity': 'Knowledge Graph', 'relationship': 'MENTIONS'}, {'entity': 'anticompetitive behavior', 'relationship': 'MENTIONS'},
{'entity': 'U.S. antitrust law', 'relationship': 'MENTIONS'}, {'entity': 'Sherman Act', 'relationship': 'MENTIONS'}, {'entity': 'Arkansas', 'relationship':
'MENTIONS'}, {'entity': 'question-and-answer technologies', 'relationship': 'MENTIONS'}, {'entity': 'Publishers', 'relationship': 'MENTIONS'}, {'entity':
'Knowledge Panel', 'relationship': 'MENTIONS'}, {'entity': 'Featured Snippets', 'relationship': 'MENTIONS'}, {'entity': 'class action antitrust suit',
'relationship': 'MENTIONS'}, {'entity': 'misappropriation of Publishers’ content', 'relationship': 'MENTIONS'}, {'entity': 'The Wall Street Journal',
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
</item 6>

<item 7>
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
</item 7>

<item 8>
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
</item 10>
=== END ITEM id=0fb87abb479bfa90 ===

=== ITEM id=79e662cede70645c ===
Gold sentences and facts:
Sentence 0: If the debt continues to rise more than the economy, risks will increase, and the government may “crowd out” private investment as it competes with the private sector for funds to service its debt.
  Fact 0.0: Risks will increase if the debt continues to rise more than the economy.
  Fact 0.1: The government competes with the private sector for funds to service its debt.
  Fact 0.2: The government may 'crowd out' private investment.
Sentence 1: And yet, “We worry about our debt-to-GDP ratio in the Philippines, as it stands at about 63% and that’s a little high for us and it is not ideal,” President Ferdinand Marcos, Jr. told members of the US-ASEAN Business Council who visited Malacañang (ABS-CBN, Aug. 9, 2023).
  Fact 1.0: President Ferdinand Marcos, Jr. said the Philippines' debt-to-GDP ratio stands at about 63%.
  Fact 1.1: President Ferdinand Marcos, Jr. said that a 63% debt-to-GDP ratio is a little high for the Philippines.
  Fact 1.2: President Ferdinand Marcos, Jr. said that a 63% debt-to-GDP ratio is not ideal.
  Fact 1.3: President Ferdinand Marcos, Jr. made the statement to members of the US-ASEAN Business Council who visited Malacañang.
  Fact 1.4: The statement is cited to ABS-CBN, Aug. 9, 2023.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
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
Graph query row:
article: A deficit budget for 2024
published_on: 2023-10-08T16:02:10+00:00
passage: A deficit budget for 2024 (Business World, 2023-10-08T16:02:10+00:00)
“After extensive deliberations and negotiations, the House of Representatives on Wednesday (Sept. 27) approved on third and final reading the P5.768-trillion
national budget for next year, just before Congress adjourns for a month-long break,” the Press and Public Affairs Bureau of Congress announced on its website,
congress.gov.ph. The 310-strong legislative chamber voted 296 against three, with zero abstentions, to approve House Bill (HB) No. 8980 or the 2024 national
budget on third and final reading. Almost unanimously.

House Speaker Ferdinand Martin G. Romualdez noted that the budgeting process was marked by rigorous discussions, particularly concerning confidential and
intelligence funds, which the House meticulously scrutinized to ensure accountability and responsible utilization. He lauded “the timely passage of the 2024
General Appropriations Bill (GAB),” stating that it not only “demonstrated the commitment of the House to upholding its constitutional role but also underscored
its dedication to serving the Filipino people with transparency and accountability.”
</item 7>

<item 8>
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
</item 8>

<item 9>
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
=== END ITEM id=79e662cede70645c ===

=== ITEM id=64edd98519562f6a ===
Gold sentences and facts:
Sentence 0: Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
  Fact 0.0: Kenneth Walker III plays for the Seahawks.
  Fact 0.1: Kenneth Walker III took four carries against the Rams.
  Fact 0.2: Kenneth Walker III left the game with an oblique injury.
Sentence 1: Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging injuries.
  Fact 1.0: Entering Week 12, Tee Higgins (hamstring) failed to suit up due to nagging injuries.
  Fact 1.1: Entering Week 12, Noah Brown (knee) failed to suit up due to nagging injuries.
  Fact 1.2: Entering Week 12, Treylon Burks (concussion) failed to suit up due to nagging injuries.
  Fact 1.3: Entering Week 12, Kadarius Toney (ankle, hip) failed to suit up due to nagging injuries.

Retrieved items:
<item 1>
Graph query row:
article: 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit
publication_date: 2023-11-19T23:51:49+00:00
publisher: CBSSports.com
connection: MENTIONS
related_entity: Kenneth Walker III
</item 1>

<item 2>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
He was seen walking on the sidelines shortly afterward but ruled doubtful to return. Rookie Zach Charbonnet, who entered Week 11 with just over 200 rushing
yards on the year, replaced Walker as Seattle's top ball carrier. Following the game, Seahawks head coach Pete Carroll told reporters Walker suffered an
"oblique strain that was legit" and could miss time.

D'Onta Foreman (Bears) remained a headliner of Chicago's backfield against the Lions, even with Khalil Herbert back from injured reserve, but managed just 14
yards on the ground before suffering an ankle injury. He walked off with a limp and attempted to run on the sidelines but did not return to the field after the
Bears ruled him questionable.

Pick Six Newsletter Crafted By The Best NFL Experts Get the day's big stories + fun stuff you love like mock drafts, picks and power rankings.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -PUBLISHED_BY-> CBSSports.com
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -PUBLISHED_BY-> CBSSports.com
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Day of Reckoning -- featuring Anthony Joshua, Deontay Wilder: Predictions, undercard, odds, picks, start time -PUBLISHED_BY-> CBSSports.com

Entities: CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit, Pete Carroll, oblique strain,
Pick Six Newsletter, Cooper Kupp, ankle injury, Lions, Rams, Seahawks, Kenneth Walker III, D'Onta Foreman, Zach Charbonnet, Bears, Khalil Herbert
</item 2>

<item 3>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
DB

Talanoa Hufanga (49ers), who emerged as a ballhawk for San Francisco's secondary in 2022, left early against the Buccaneers with a knee injury. Coach Kyle
Shanahan confirmed to reporters on Monday that the All-Pro suffered a season-ending ACL tear.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -PUBLISHED_BY-> CBSSports.com
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -PUBLISHED_BY-> CBSSports.com
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -PUBLISHED_BY-> CBSSports.com
- Day of Reckoning -- featuring Anthony Joshua, Deontay Wilder: Predictions, undercard, odds, picks, start time -PUBLISHED_BY-> CBSSports.com

Entities: CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit, Cooper Kupp, Talanoa Hufanga,
ACL tear, knee injury, Rams, Seahawks, Kenneth Walker III, San Francisco, Kyle Shanahan, Buccaneers, 49ers
</item 3>

<item 4>
Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver (Sporting News, 2023-11-29T20:27:00+00:00)
Geno Smith's struggles complicate their fantasy prospects, as well.

What are the biggest WR injuries in Week 13?

Entering Week 12, Tee Higgins (hamstring), Noah Brown (knee), Treylon Burks (concussion), and Kadarius Toney (ankle, hip) all failed to suit up due to nagging
injuries. For now, we've placed Higgins and Brown in our Week 13 list and omitted Burks and Toney. We'll also be eyeing the Chargers' Josh Palmer (knee), who's
eligible to come off the IR this week but probably won't suit up against New England.

Last week, the most notable WRs to exit their respective games with injuries include Chris Olave (concussion), Amari Cooper (ribs), Demario Douglas
(concussion), and Rashid Shaheed (thigh). We've left all four WRs out of our initial Week 13 list but will closely monitor their teams' injury reports to gauge
their playing prospects.

Graph facts:
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Chris Olave
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Chris Olave
- Fantasy Football WR PPR Rankings Week 14: Who to start, best sleepers at wide receiver -MENTIONS-> Chris Olave
- Chris Olave -PARTICIPATED_IN-> Jaguars vs. Saints on Week 7 "Thursday Night Football"
- Chris Olave -PARTICIPATED_IN-> Jaguars vs. Saints game

Entities: Chris Olave, New England, Josh Palmer, Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver, Demario Douglas, Rashid
Shaheed, Amari Cooper, Noah Brown, Treylon Burks, Kadarius Toney, Sporting News, Tee Higgins, Geno Smith, Chargers
</item 4>

<item 5>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Smith did return, however, for the final drive of the game in the losing effort. Head coach Pete Carroll said that Smith will "make it back." When asked if that
means for Thursday's matchup against San Francisco, the coach said, "I would think so."

Sam Howell (Commanders) received medical attention on the sidelines after enduring several hard hits to the upper body on a touchdown run against the Giants. He
was quickly cleared of a potential concussion, however, and remained in the game. Veteran journeyman Jacoby Brissett is Washington's backup.

RB

Aaron Jones (Packers) suited up against the Chargers despite a lingering hamstring issue, only to manage just five carries before suffering a knee injury. The
former Pro Bowler needed help to leave the field and was then carted to the locker room, replaced by A.J. Dillon.
</item 5>

<item 6>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Week 11 of the 2023 NFL season has provided plenty of drama, from the Bears hanging with the Lions to the Giants getting a rebound from emergency quarterback
Tommy DeVito. But the latest round of regular-season action has also brought another wave of unfortunate injuries. Here's a roundup of the latest big names to
go down around the league:

QB

C.J. Stroud (Texans) left for a single play against the Cardinals after absorbing a hard hit, with backup Davis Mills entering in relief. Stroud did not receive
an injury designation, however, and returned after a quick check-in from trainers.

Geno Smith (Seahawks) suffered an injury to his throwing arm after guiding a 16-7 lead over the Rams, showing frustration while trying to stretch the arm after
a hit from Aaron Donald. Seattle quickly ruled him questionable to return with an elbow issue, and backup Drew Lock took over under center with 18 seconds left
in the third quarter.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL
- NFL DFS picks Week 15: Best sleepers, value players for FanDuel, DraftKings, SuperDraft lineups -MENTIONS-> Tommy DeVito
- Fantasy Football QB Rankings Week 14: Who to start, best sleepers at quarterback -MENTIONS-> Tommy DeVito
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Tommy DeVito
- Brock Purdy -MENTIONS-> NFL

Entities: Tommy DeVito, NFL, CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit, Davis
Mills, Cooper Kupp, Texans, Cardinals, C.J. Stroud, Lions, Rams, Geno Smith, Drew Lock, Seahawks, Kenneth Walker III
</item 6>

<item 7>
NFL Week 13 injuries: Browns to start Joe Flacco with Dorian Thompson-Robinson out, Saints' Olave questionable (CBSSports.com, 2023-12-01T22:52:36+00:00)
Bengals at Jaguars (-8.5), Monday

Bengals : CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT

: CB Cam Taylor-Britt (quad/ankle), D'Ante Smith (knee), Jay Tufele (illness) OUT Jaguars: CB Tyson Campbell (hamstring), RB Travis Etienne (ribs), TE Brenton
Strange (foot), TE Luke Farrell (toe) QUESTIONABLE

The Bengals got good news on the injury front ahead of Monday's game against the Jaguars. Wide receiver Tee Higgins (hamstring/ankle) and linebacker Logan
Wilson (ankle/foot) are set to play, according to head coach Zac Taylor. Higgins and Wilson were full participants in practice. Cornerback Cam Taylor-Britt hurt
his foot in practice and is out.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Jaguars
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -MENTIONS-> Jaguars
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jaguars
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jaguars
- Jaguars vs. Saints live score, updates, highlights from NFL 'Thursday Night Football' game -MENTIONS-> Jaguars

Entities: Jaguars, Saints, Tyson Campbell, Bengals, CBSSports.com, injuries, Olave, NFL Week 13 injuries: Browns to start Joe Flacco with Dorian
Thompson-Robinson out, Saints' Olave questionable, Browns, Dorian Thompson-Robinson, D'Ante Smith, Jay Tufele, Brenton Strange, Luke Farrell, Tee Higgins
</item 7>

<item 8>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
I agree to receive the "Pick Six Newsletter" and marketing communications, updates, special offers (including partner offers), and other information from CBS
Sports and the Paramount family of companies. By pressing sign up, I confirm that I have read and agree to the Terms of Use and acknowledge Paramount's Privacy
Policy See All Newsletters Please check the opt-in box to acknowledge that you would like to subscribe. Thanks for signing up! Keep an eye on your inbox. Sorry!
There was an error processing your subscription.

WR

Tyreek Hill (Dolphins) briefly left Miami's game against the Raiders while nursing a hand injury and was ruled questionable to return, but he took the field
again soon afterward, securing seven catches for 80 yards and a score before halftime. Hill has left multiple games this year, only to return in all of them.

Cooper Kupp (Rams) had just a single catch against the Seahawks before suffering an ankle injury while blocking.

Graph facts:
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Raiders
- Kelce -MENTIONS-> Raiders
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Raiders
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> Raiders
- Fantasy Football WR PPR Rankings Week 13: Who to start, best sleepers at wide receiver -MENTIONS-> Raiders

Entities: Raiders, hand injury, game against the Seahawks, CBSSports.com, 2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III
among stars to exit, Cooper Kupp, Miami's game against the Raiders, Dolphins, ankle injury, NFL, Rams, Seahawks, Tyreek Hill
</item 8>

<item 9>
2023 NFL Week 11 injury roundup: Rams WR Cooper Kupp, Seahawks RB Kenneth Walker III among stars to exit (CBSSports.com, 2023-11-19T23:51:49+00:00)
Dillon. Green Bay ruled him questionable to return, then downgraded him to out, but coach Matt LaFleur said after the Packers' win over Los Angeles that he
doesn't expect the injury to be a long-term issue.

De'Von Achane (Dolphins) returned from injured reserve -- and a four-game absence -- but limped to the sidelines after his second touch of the first quarter
against the Raiders. Miami ruled him questionable to return with a knee injury, and while he later returned to the sidelines from a locker-room visit, he was
replaced on the field indefinitely by Raheem Mostert. Head coach Mike McDaniel told reporters on Monday that Achane is day-to-day. While he hasn't been ruled
out of Friday's matchup with the Jets, he's not been given the green light either, so his status is worth monitoring.

Kenneth Walker III (Seahawks) took four carries against the Rams before leaving with an oblique injury.
</item 9>

<item 10>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
They weren't awful against Dallas last week on a per-touch basis, but neither finished with more than 12 touches. Even worse, no Carolina back has enjoyed a
60-yard rushing game since Oct. 15. Bad offensive line, negative game scripts, awful coaching — it's all bad, man!

What are the biggest RB injuries in Week 12?

Second-year Seahawks back Kenneth Walker III suffered an oblique injury in Week 11 that the ever-hip Pete Carroll referred to as "legit." KW3 has a better
chance of missing multiple weeks than he does at suiting up for Seattle's Thanksgiving Day game against Fred Warner and the 49ers. If Zach Charbonnet somehow
finds himself on your waiver wire, an early Merry Christmas to you if you have the FAAB or a high waiver priority.

The hits just keep on coming for the Packers' Aaron Jones, who suffered a knee injury but seems to have at least caught a break in the sense that he didn't tear
his ACL.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Packers
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Packers
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Packers
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Packers
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Packers

Entities: Packers, Pete Carroll, Sporting News, knee injury, Seahawks, Kenneth Walker III, Aaron Jones, Zach Charbonnet, oblique injury, ACL, Fantasy Football
RB Rankings Week 12: Who to start, best sleepers at running back, 49ers, Fred Warner
</item 10>
=== END ITEM id=64edd98519562f6a ===
