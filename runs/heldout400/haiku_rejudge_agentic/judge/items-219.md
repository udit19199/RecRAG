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

=== ITEM id=eb2fcaa0e99bfff5 ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: Google makes deals with Apple.
  Fact 0.2: Google makes deals with Samsung.
  Fact 0.3: Google makes deals with Mozilla.
  Fact 0.4: Google makes deals with many others.
  Fact 0.5: The deals are meant to ensure Google is the default search engine on practically every platform.
Sentence 1: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 1.0: This will probably come up later in the trial.
  Fact 1.1: Epic chose to focus more on simply painting Google as the bad guy.
  Fact 1.2: Epic did this on day one.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case says this is done through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, Supreme Court, Epic, Google, V-Bucks, Macintosh, Apple trial, Google lawsuit, Google trial, Apple's “1984” Macintosh ad, Epic v. Google,
explained, Fortnite, The Verge
</item 2>

<item 3>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Google Play
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Google Play
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Google Play
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Google Play
- Beeper Mini is back in operation after Apple’s attempt to shut it down -MENTIONS-> Google Play

Entities: Google Play, Activision, Epic, Google, The Epic v. Google trial may come down to simple v. complicated, app store and payment processing fees, Epic v.
Google trial, competition, The Verge
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
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
The competitive landscape with other app stores, OEMs

The tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does
the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app
stores like Samsung’s Galaxy Store and Amazon’s Appstore.

Epic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are
still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment
manufacturers) have to agree to in order to use Android.

Epic will also bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps
outside of Google Play.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Google Play
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Google Play
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Google Play
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Google Play
- Beeper Mini is back in operation after Apple’s attempt to shut it down -MENTIONS-> Google Play

Entities: Epic Games’ antitrust battle with Google, Google Play, App Store, Android, Galaxy Store, Amazon Appstore, Android OEMs, Google apps,
Anti-Fragmentation Agreements, Ninth Circuit, Ninth Circuit ruling in the Apple case, App store competition, What to know about Fortnite maker Epic Games’
antitrust battle with Google, starting today, Epic Games, Apple
</item 5>

<item 6>
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
Graph query row:
company: Amazon antitrust lawsuit
page: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
passage: More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 8>

<item 9>
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
</item 9>

<item 10>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android

Entities: Android, Apple, Epic, Google, The Epic v. Google trial may come down to simple v. complicated, Pomerantz, Project Liberty, Android sideloading flow,
app store fees, Google policy, app stores, Epic v. Google trial, Fortnite, The Verge
</item 10>
=== END ITEM id=eb2fcaa0e99bfff5 ===

=== ITEM id=e12d1cb8b798ef29 ===
Gold sentences and facts:
Sentence 0: Because Flexport makes money by charging fees for the transportation of goods, the company's business was getting hammered.
  Fact 0.0: Flexport makes money by charging fees for the transportation of goods.
  Fact 0.1: Flexport's business was getting hammered.
Sentence 1: Petersen has spent the past month cutting costs, including laying off about 20% of its workers, or about 600 people.
  Fact 1.0: Petersen has spent the past month cutting costs.
  Fact 1.1: Petersen has laid off about 20% of Flexport's workers.
  Fact 1.2: Petersen's layoffs were about 600 people.
Sentence 2: Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board, calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
  Fact 2.0: There was turmoil at Flexport.
  Fact 2.1: Dave Clark is a former Amazon executive.
  Fact 2.2: Dave Clark was ousted as CEO of Flexport just a year into the job.
  Fact 2.3: Dave Clark fired back at Flexport's founder and board.
  Fact 2.4: Dave Clark called recent reporting on Flexport 'deeply concerning'.
  Fact 2.5: Dave Clark made the comments on Monday in a lengthy post on the social media site X.
  Fact 2.6: Dave Clark's post followed a CNBC report.
  Fact 2.7: The CNBC report provided new information about Dave Clark's last days at Flexport.
  Fact 2.8: Flexport is a freight forwarding and customs brokerage startup.
  Fact 2.9: Flexport is valued at $8 billion.

Retrieved items:
<item 1>
Graph query row:
article: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier
published_on: 2023-10-07T20:15:26+00:00
publisher: TechCrunch
mentioned_entities: ['SBF Trial', 'Google’s hardware event', 'Pixel 8 Pro', 'Flexport drama', 'FTX fallout', 'Gmail', 'ad-free subscription plan', 'Induced AI',
'Manish', 'Google’s annual hardware event', 'Pixel Fold', 'Pixel Buds', 'Pixel Watch 2', 'teens’ startup', 'ad-free tier', 'Pixel 7 Pro', 'recent reporting on
Flexport', 'Ingrid', 'Recruiter', 'LinkedIn Learning coach', 'AI-powered tool for marketing campaigns', 'TikTok’s ad-free plan', 'X users’ daily posts and
content generation', 'X users’ daily posts and content generation figures', 'Pixel 8', 'IronNet', 'Keith Alexander', 'NSA', 'going public', 'blockchain',
'Equity', 'At One Ventures', 'Section 32', 'Greylock', 'Fearless Fund', 'grants', 'Black women founders', 'Found', 'Esther Rodriguez-Villegas', 'Acurable',
'patient-friendly wearable devices', 'respiratory conditions', 'epilepsy', 'Chain Reaction', 'Jacquelyn', 'Southern District of New York courthouse',
'Bankman-Fried', 'SBF and FTX trial', 'Breathe Battery Technologies', 'software', 'lithium-ion battery', 'faster charging speeds', 'greater longevity', 'Anna',
'10 investors', 'survey of 10 investors', 'oligopoly dynamics', 'first-mover advantages', 'proprietary data', 'Linda Yaccarino', 'Darrell', 'ChatGPT', 'AI
tools', 'LinkedIn', 'Rainforest', 'Virginia', 'TechCrunch+', 'Tim', 'the defense', 'Bard', 'X', 'Christine', 'AI', 'spam', 'Elon Musk', 'startups', 'FTX',
'Google', 'Rebecca', 'the prosecution', 'TikTok', 'cybersecurity', 'CNBC', 'venture funding', 'sleep apnea', 'Alex', 'Sam Altman', 'Google Assistant', 'Android
14', 'U.S.', 'fraud', 'Dave Clark', 'Flexport', 'Amazon']
passages: ['Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nHiya, folks,
and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the past few
days, and — we humbly submit — it’s a darn useful resource for folks on the go.\n\nThis week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware
event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok
testing an ad-free subscription plan, and LinkedIn going big on AI tools. And that’s not all.\n\nIf you haven’t, sign up here to get WiR in your inbox every
Saturday. And if you have, our thanks. Now, let’s get on with the news.', 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an
ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nAnd if you have, our thanks. Now, let’s get on with the news.\n\nMost read\n\nAltman backs teen
entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating workflows in
“previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and converts the
instructions to pseudo-code in real time.\n\nGoogle unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant
showed off what it’s been working on. Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold,
Android 14, Pixel Buds, Google Assistant, Bard, Pixel Watch 2 and other goodies.', 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests
an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nHands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked
what he saw. While very similar to last year’s model (the Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features
made it enough of an upgrade to (potentially) warrant a purchase — minus the underutilized temperature sensor. Stay tuned for his full review.\n\nTurmoil at
Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board, calling
recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a report from
CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.', 'Sam Altman backs
teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nFor $4.99, subscribers get an ad-free
experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that it’s piloting the plan in a single,
English-speaking market outside the U.S. for now.\n\nLinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job
hunting, marketing and sales products, Ingrid writes. They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it
throughout; an AI-powered LinkedIn Learning coach; and a new AI-powered tool for marketing campaigns.\n\nMusk comes clean about X’s metrics — maybe: In
September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million posts every day, excluding retweets. But speaking
at an event this week, X CEO Linda Yaccarino offered a contradictory figure.', 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an
ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nShe claimed X was seeing 500 million posts per day on the platform. So who’s right? Beats us.\n\nFormer
NSA director’s startup shutters: IronNet, a once-promising cybersecurity startup founded by a former NSA director, has shuttered and laid off its remaining
staff following its collapse. The Virginia-based IronNet was founded in 2014 by retired four-star general Keith Alexander and had raised more than $400 million
in funding. But IronNet failed to gain traction after going public in August 2021, and its stock price continued to steeply decline in the wake of an initial
spike.\n\nAudio\n\nOn the hunt for a new podcast to listen to while you work out, do the dishes or rake the leaves (now that fall’s arrived)? Look no further
than TechCrunch’s roster, which covers the world of startups, the blockchain and more.', 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok
tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nOn Equity this week, the crew talked about the SBF trial; deals from VR firms Rainforest, At One
Ventures, Section 32 and Greylock, where venture funding has declined; and how Fearless Fund, a firm founded to invest in women of color, is being barred from
awarding grants to Black women founders.\n\nMeanwhile, Found featured Esther Rodriguez-Villegas from Acurable, a medical device company that makes
patient-friendly wearable devices to diagnose and manage respiratory conditions at home. As a career-long academic, Rodriguez-Villegas talks about how she never
intended to be a founder until she learned about how the currently available medical devices make it extremely difficult to detect and treat diseases like sleep
apnea and epilepsy.\n\nAnd over on Chain Reaction, Jacquelyn did a crossover episode with Alex about the SBF trial.', 'Sam Altman backs teens’ startup, Google
unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)\nJacquelyn has been on the ground at the Southern District of New
York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots to talk about.\n\nTechCrunch+\n\nTC+ subscribers get access to
in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up. Here are a few highlights from
this week:\n\nInside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone
who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to
commit fraud or steal and basically got in over his head.', 'Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier
(TechCrunch, 2023-10-07T20:15:26+00:00)\nBattery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software
that can be slipped into just about any lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.\n\nWhat lies
beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched on where
startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.']
</item 1>

<item 2>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Groupon

Impacting another 500 employees announced on January 31. The company said this new set of layoffs will be spread across the first two quarters of 2023.

Impossible Foods

Reportedly affecting 20% of its staff, over 100 employees, Bloomberg reported first.

PayPal

Announced on January 30, about 2,000 full-time employees, or 7% of its workforce, were affected.

Arrival

Announced on January 30, with a newly appointed CEO, slashing 50% of its workforce — 800 employees globally.

Waymo

The self-driving technology unit under Alphabet quietly laid off workers on January 24, according to The Information and several posts on LinkedIn and Blind.
It’s not yet clear how many of Waymo’s staff will be affected.

Spotify

Announced on January 23, impacting around 6% of its global workforce — around 600 employees.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> PayPal
- Here’s how Rainforest, a budding Stripe rival, aims to win over software companies -MENTIONS-> PayPal
- Discogs’ vibrant vinyl community is shattering -MENTIONS-> PayPal
- Thiel -WORKS_FOR-> PayPal
- Hoffman -WORKS_FOR-> PayPal

Entities: PayPal, Alphabet, tech layoffs, A comprehensive list of 2023 tech layoffs, The Information, Groupon, Impossible Foods, Arrival, Waymo, Blind, Groupon
layoffs, Impossible Foods layoffs, PayPal layoffs, Arrival layoffs, Waymo layoffs
</item 2>

<item 3>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Superpedestrian

Is shutting down its U.S.-based shared scooter operation and exploring a sale of its European business, TechCrunch has exclusively learned.

Bolt

Confirmed it laid off 29% of its staff December 15. The e-commerce company known for its one-click checkout technology was at one time the subject of a federal
probe.

Cruise

In an announcement obtained exclusively by TechCrunch, the embattled self-driving car company is cutting 900 employees, or 24% of its workforce, per a December
14 email from new president and CTO Mo Elshenawy.

Etsy

Is laying off 11% of its workforce. In a letter to employees December 13, CEO Josh Silverman cited a “very challenging macro and competitive environment.”

Chipper Cash

Has reportedly laid off 15 people across various departments. It would be the fintech unicorn’s fourth round of layoffs this year.

SmileDirectClub

Has shut down three months after filing for Chapter 11 bankruptcy, citing $900 million of debt.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Chapter 11 bankruptcy
- Alameda -PARTICIPATED_IN-> Chapter 11 bankruptcy
- FTX -PARTICIPATED_IN-> Chapter 11 bankruptcy
- Chapter 11 bankruptcy -LOCATED_IN-> U.S.
- 2023 in social media: the case for the fediverse -MENTIONS-> Etsy

Entities: Chapter 11 bankruptcy, Superpedestrian, Etsy, SmileDirectClub, Mo Elshenawy, Josh Silverman, federal probe, A comprehensive list of 2023 tech layoffs,
Chipper Cash, Bolt, TechCrunch, Cruise
</item 3>

<item 4>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Hippo

Plans to eliminate 120 roles, the company said in an SEC filing October 30.

Liberty Mutual

Plans to cut 850 members of its staff, the insurance company confirmed October 27.

Salsify

Cut 110 jobs and are moving some roles abroad, CEO and co-founder Jason Purcell announced October 25.

Slync

Is shutting down, months after its former CEO Chris Kirchner was arrested on fraud charges and sued the company to pay for his legal bills.

Shipt

Announced October 24 in a company statement that it laid off 3.5% of their workforce and will close many open roles.

Pebble

The X alternative previously known as T2 announced October 24 that it is shutting down its operations.

Nokia

Announced October 19 that it will cut up to 14,000 jobs, or 16% of its workforce, following a sizable dip in Q3.

Roblox China

Cut 15 roles on October 24, two years after the joint effort with Tencent abruptly paused its service.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Turkey
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Yahoo
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg

Entities: A comprehensive list of 2023 tech layoffs, Hippo, Salsify, Slync, Shipt, Nokia, Roblox China, Jason Purcell, Chris Kirchner, fraud charges, Hippo job
cuts, Liberty Mutual staff cuts, Salsify job cuts, Shipt workforce layoffs, Nokia job cuts
</item 4>

<item 5>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
The company is now facing an unfair labor practice complaint, with workers alleging the layoffs are retaliation for unionizing.

Chewy

Laid off more than 200 employees November 14. The cuts impact roles in HR, recruiting, data, business intelligence, plus directors and higher managers.

Amazon

Is cutting “just over 180” roles in its gaming division, VP Christoph Hartmann wrote in an email to employees November 13.

Carta

Is reportedly cutting more of its workforce. The number of positions affected are unknown, but it would be the unicorn startup’s third layoff round this year.

Pico

Is cutting a “few hundred” roles, a person familiar with the matter told TechCrunch November 9. ByteDance’s Oculus challenger held an internal meeting
announcing it is halting its expansion this week.

Cruise

Started laying off contingent workers who supported driverless operations November 9. Those layoffs came after the company recalled 950 robotaxis and lost key
commercial permits to operate in California.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Turkey
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Yahoo
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg

Entities: A comprehensive list of 2023 tech layoffs, Chewy, Carta, Pico, Christoph Hartmann, Oculus, unfair labor practice complaint, unionizing, driverless
operations, Chewy layoffs, Amazon gaming division cuts, Pico cuts, Cruise contingent-worker layoffs, Pico internal meeting, Cruise recall of 950 robotaxis
</item 5>

<item 6>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
VinFast

VinFast has not shared how many employees have been cut, but a LinkedIn post from a former employee said “nearly 35 roles” were affected. Announced on February
6.

Dell

Announced February 6, impacting 6,650 people, or 5% of worldwide workforce.

Getaround

Announced February 2, 10% of staff — about 42 employees.

Pinterest

Announced February 2, 150 employees impacted. This is the second job-cutting move within weeks of the first round in December 2022.

Rivian

Announced on February 1, cutting 6% of its workforce for the second time in less than a year.

January 2023

SoFi Technologies

Announced on January 31, cutting 65 jobs, or about 5% of its 1,300-person workforce. First reported by The Wall Street Journal.

NetApp

Announced on January 31, impacting 8% of its staff — about 960 people.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> The Wall Street Journal
- report on Thursday -PUBLISHED_BY-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy

Entities: The Wall Street Journal, tech layoffs, A comprehensive list of 2023 tech layoffs, VinFast, Getaround, Pinterest, SoFi Technologies, NetApp, VinFast
layoffs, Dell layoffs, Getaround layoffs, Pinterest layoffs, Rivian layoffs, SoFi Technologies layoffs, NetApp layoffs
</item 6>

<item 7>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Future of flight

EVTOL company Archer Aviation plans to start air taxi operations in Abu Dhabi in 2026. From there, Archer says it will expand across the UAE as part of a
memorandum of understanding with the Abu Dhabi Investment Office.

Miscellaneous

The United Auto Workers strike is affecting CES. Stellantis canceled its planned presentations for the tech trade show in January, citing the cost of ongoing
UAW strikes.

Zipcar is getting hit with a fine from the NHTSA for “renting vehicles with open, unrepaired recalls.” One recall concerns 2015–2017 Ford Transit Vans, which
continued to appear on Zipcar’s platform despite safety issues with the vehicles.

People

Autobrains hired Uri Yacovy, a former SVP at Mobileye, as its chief operating officer.

Logistics company Flexport is laying off 20% of workers, or about 600 people, topping off a spate of staff upheavals at the company.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Flexport layoffs
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch

Entities: Flexport layoffs, TechCrunch, CES, Abu Dhabi, Rivian, Rebelle, Convoy, Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian
scores a win at Rebelle, NHTSA, Archer Aviation, UAE, Abu Dhabi Investment Office, United Auto Workers, United Auto Workers strike, Zipcar
</item 7>

<item 8>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Chipper Cash relieved almost one-third of its workforce, about 100 employees.

Evernote

On February 17 the company confirmed laying off 129 people.

Jumia

Announced February 16 that it cut 20% of its staff, or more than 900 positions across its 11 markets, in Q4 of 2022.

Convoy

Announced on February 16 that it is shuttering its Atlanta office and laying off workers as part of restructuring. This is the third time in less than a year
that the company has laid off workers.

Sprinklr

Announced on February 15 that it will impact 4% of its global workforce — or more than 100 employees.

iRobot

Announced on February 13 that it will lay off 7% of its workforce, roughly 85 employees.

Twilio

Announced on February 13 that it will impact around 17% of its global workforce, about 1,400 people.

Graph facts:
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> iRobot
- iRobot -MENTIONS-> intense competition
- The 26 best Amazon Prime Day deals you can still shop for today -MENTIONS-> iRobot
- Amazon -SAID-> iRobot
- iRobot -PARTICIPATED_IN-> Amazon’s iRobot purchase

Entities: iRobot, tech layoffs, A comprehensive list of 2023 tech layoffs, Chipper Cash, relieved almost one-third of its workforce, about 100 employees,
Evernote, confirmed laying off 129 people, Jumia, cut 20% of its staff, or more than 900 positions across its 11 markets, in Q4 of 2022, shuttering its Atlanta
office, laying off workers as part of restructuring, Sprinklr, impact 4% of its global workforce, or more than 100 employees, lay off 7% of its workforce,
roughly 85 employees, Twilio
</item 8>

<item 9>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Prior to the layoff, TuSimple had about 550 employees in the U.S. and post reduction in force the company will have about 220, according to the company.

Nuro

Announced May 12 that it will lay off 30%, or about 340 employees across the company.

LinkedIn

Announced May 8 that it is cutting 716 jobs, or about 3.6% of total employees, and will phase out its local jobs app in China. Despite reducing some roles,
LinkedIn also plans to open about 250 new jobs on May 15.

Rapid

Announced May 5 that Rapid (previously known as RapidAPI) has laid off another 70 employees less than two weeks after letting go of 50% of its staff. Just 42
people remain at the company, down from 230 in April, dropping a total of 82% in headcount.

Meesho

Announced May 4 that it has cut 15% of its workforce, or 251 roles.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Turkey
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Yahoo
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg

Entities: A comprehensive list of 2023 tech layoffs, LinkedIn, TuSimple, Nuro, local jobs app, Rapid, Meesho, TechCrunch, China, U.S.
</item 9>

<item 10>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Bandcamp

Eliminated 50% of its staff on October 16 after ownership changed from video game company Epic to music licensing platform Songtradr.

LinkedIn

Confirmed October 16 that it will cut 668 more jobs, bringing the total to nearly 1,400 this year, with the bulk of the most recent cuts impacting those with
R&D roles.

Flexport

On October 13 began laying off 20% of its workers, about 600 people, as it prepares for restructuring, and continues a cost-cutting campaign that began with
rescinding offers and working to lease office space.

Qualcomm

Is cutting 1,258 jobs in two of its California offices. According to a filing with the California Employment Development Department, the semiconductor company
plans to make the reductions in December.

Blue Origin

Laid off 40 people October 10, including software engineers and program managers, according to The Information.

Stitch Fix

Will lay off 558 employees after the personal styling subscription service closes a distribution center in Dallas.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Turkey
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Yahoo
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg

Entities: A comprehensive list of 2023 tech layoffs, Bandcamp, Songtradr, Bandcamp staff layoffs, LinkedIn, LinkedIn job cuts, Flexport layoffs, Qualcomm,
Qualcomm job cuts, California Employment Development Department, Blue Origin, Blue Origin layoffs, The Information, Stitch Fix, Stitch Fix layoffs
</item 10>
=== END ITEM id=e12d1cb8b798ef29 ===

=== ITEM id=fc04bfae5fcc71ad ===
Gold sentences and facts:
Sentence 0: In the year since Altman catapulted ChatGPT to global fame, he has become Silicon Valley’s sought-after voice on the promise and potential dangers of artificial intelligence and his sudden and mostly unexplained exit brought uncertainty to the industry’s future.
  Fact 0.0: In the year since Altman catapulted ChatGPT to global fame, Altman has become Silicon Valley's sought-after voice on the promise of artificial intelligence.
  Fact 0.1: In the year since Altman catapulted ChatGPT to global fame, Altman has become Silicon Valley's sought-after voice on the potential dangers of artificial intelligence.
  Fact 0.2: Altman catapulted ChatGPT to global fame.
  Fact 0.3: Altman's exit was sudden and mostly unexplained.
  Fact 0.4: Altman's sudden and mostly unexplained exit brought uncertainty to the industry's future.
Sentence 1: This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout.
  Fact 1.0: This week's coverage includes Sam Altman backing a teen's AI startup.
  Fact 1.1: This week's coverage includes Google's hardware event.
  Fact 1.2: The Google hardware event coverage includes first impressions of the Pixel 8 Pro.
  Fact 1.3: This week's coverage includes Flexport drama.
  Fact 1.4: This week's coverage includes the ongoing FTX fallout.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The board's language is a basis for a theory.
  Fact 2.1: The way giant tech companies work is a basis for a theory.
  Fact 2.2: The theory is the prevailing theory floating around right now.

Retrieved items:
<item 1>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
The company said its board consists of OpenAI’s chief scientist, Ilya Sutskever, and three non-employees: Quora CEO Adam D’Angelo, tech entrepreneur Tasha
McCauley, and Helen Toner of the Georgetown Center for Security and Emerging Technology.

OpenAI’s key business partner, Microsoft, which has invested billions of dollars into the startup and helped provide the computing power to run its AI systems,
said that the transition won’t affect its relationship.

“We have a long-term partnership with OpenAI and Microsoft remains committed to Mira and their team as we bring this next era of AI to our customers,” said an
emailed Microsoft statement.

While not trained as an AI engineer, Altman, now 38, has been seen as a Silicon Valley wunderkind since his early 20s. He was recruited in 2014 to take lead of
the startup incubator YCombinator.

Graph facts:
- Microsoft -MENTIONS-> device-agnostic ecosystem
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Microsoft
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Microsoft
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Microsoft
- Microsoft -MENTIONS-> cloud gaming

Entities: Microsoft, AI, Fortune, Silicon Valley, OpenAI, Ilya Sutskever, OpenAI board, Quora, Adam D’Angelo, Tasha McCauley, Georgetown Center for Security and
Emerging Technology, YCombinator, OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board
did', Mira, Helen Toner
</item 1>

<item 2>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He was recruited in 2014 to take lead of the startup incubator YCombinator.

“Sam is one of the smartest people I know, and understands startups better than perhaps anyone I know, including myself,” read YCombinator co-founder Paul
Graham’s 2014 announcement that Altman would become its president. Graham said at the time that Altman was “one of those rare people who manage to be both
fearsomely effective and yet fundamentally benevolent.”

OpenAI started out as a nonprofit when it launched with financial backing from Tesla CEO Elon Musk and others. Its stated aims were to “advance digital
intelligence in the way that is most likely to benefit humanity as a whole, unconstrained by a need to generate financial return.”

That changed in 2018 when it incorporated a for-profit business Open AI LP, and shifted nearly all its staff into the business, not long after releasing its
first generation of the GPT large language model for mimicking human writing.

Graph facts:
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Elon Musk
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> Elon Musk
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> Elon Musk
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Elon Musk
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Elon Musk

Entities: Elon Musk, Fortune, OpenAI, firing Altman, YCombinator, Paul Graham, Open AI LP, GPT large language model, OpenAI's ex-chairman accuses board of going
rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did', Board, Sam Altman, Tesla
</item 2>

<item 3>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
In perhaps the most unexpected tech news of the year, billionaire and AI evangelist Sam Altman has been ejected from his CEO role at OpenAI by the company’s
board after an apparent vote of no confidence. Its exact wording in a release issued this afternoon: Altman’s “departure follows a deliberative review process
by the board, which concluded that he was not consistently candid in his communications with the board, hindering its ability to exercise its responsibilities.”

What the hell is happening at the most hyped company in the world?! Here are some totally speculative theories that occurred to us and others around the web.

1. Did Altman circumvent the board in a major deal?

Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now. “Not consistently candid” is
a very diplomatic way of saying Altman lied.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, OpenAI, OpenAI's board, vote of no confidence, WTF is going on at OpenAI? We have theories, Sam Altman
</item 3>

<item 4>
Graph query row:
article: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier
publisher: TechCrunch
entity: Sam Altman
relationship: MENTIONS
</item 4>

<item 5>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He’s sat with multiple heads of state to discuss AI’s potential and perils. Just Thursday, he took part in a CEO summit at the Asia-Pacific Economic Cooperation
conference in San Francisco, where OpenAI is based.

He predicted AI will prove to be “the greatest leap forward of any of the big technological revolutions we’ve had so far.” He also acknowledged the need for
guardrails, calling attention to the existential dangers future AI could pose.

Some computer scientists have criticized that focus on far-off risks as distracting from the real-world limitations and harms of current AI products. The U.S.
Federal Trade Commission has launched an investigation into whether OpenAI violated consumer protection laws by scraping public data and publishing false
information through its chatbot.

Graph facts:
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> false information
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Altman
- How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement -MENTIONS-> Altman
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> Altman
- WTF is going on at OpenAI? We have theories -MENTIONS-> Altman

Entities: false information, Altman, consumer protection laws, AI, Fortune, San Francisco, OpenAI, CEO summit, Asia-Pacific Economic Cooperation conference,
U.S. Federal Trade Commission, investigation into whether OpenAI violated consumer protection laws by scraping public data and publishing false information
through its chatbot, public data, chatbot, OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the
board did'
</item 5>

<item 6>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Altman and others pushed back, Musk quit and said he would focus on his own AI work at Tesla. In February 2018, he announced his departure to OpenAI’s
staff on the top floor of the startup’s offices in a converted truck factory, three people who attended the meeting said. When he said that OpenAI needed to
move faster, one researcher retorted at the meeting that Musk was being reckless. Musk called the researcher a “jackass” and stormed out, taking his deep
pockets with him. OpenAI suddenly needed new financing in a hurry. Altman flew to Sun Valley for a conference and ran into Satya Nadella, Microsoft’s CEO. A
tie-up seemed natural. Altman knew Microsoft’s chief technology officer, Kevin Scott. Microsoft had bought LinkedIn from Hoffman, an OpenAI board member.
Nadella told Scott to get it done. The deal closed in 2019.

Graph facts:
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> Musk
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Musk
- Founders, are events useful? -MENTIONS-> Musk
- X (formerly Twitter) is worst for disinformation, per EU analysis -MENTIONS-> Musk
- The UK just laid out new rules for the internet — it only gets harder from here -MENTIONS-> Musk

Entities: Musk, Satya Nadella, conference, How ego and fear fuelled the rise of artificial intelligence, Hoffman, Altman, Sun Valley, Kevin Scott, The deal,
LinkedIn, Microsoft, The Age, artificial intelligence, OpenAI, Tesla
</item 6>

<item 7>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
AI was once laughed off. Now the richest men in Silicon Valley were shelling out billions to keep from being left behind. The lost ethics board

When Musk invested in DeepMind, he broke his own informal rule – that he would not invest in any company he didn’t run himself. The downsides of his decision
were already apparent when, only a month or so after his birthday spat with Page, he again found himself face to face with his former friend and fellow
billionaire. Larry Page, co-founder of Google. Credit: Bloomberg The occasion was the first meeting of DeepMind’s ethics board, on August 14, 2015. The board
had been set up at the insistence of the startup’s founders to ensure that their technology did no harm after the sale. The members convened in a conference
room just outside Musk’s office at SpaceX. But that’s where Musk’s control ended. When Google bought DeepMind, it bought the whole thing. Musk was out.

Graph facts:
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> Musk
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Musk
- Founders, are events useful? -MENTIONS-> Musk
- X (formerly Twitter) is worst for disinformation, per EU analysis -MENTIONS-> Musk
- The UK just laid out new rules for the internet — it only gets harder from here -MENTIONS-> Musk

Entities: Musk, SpaceX, How ego and fear fuelled the rise of artificial intelligence, Larry Page, DeepMind ethics board, first meeting of DeepMind’s ethics
board, birthday spat, DeepMind, The Age, Google, Silicon Valley, artificial intelligence, Bloomberg
</item 7>

<item 8>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Around the same time, Musk, who had co-chaired its board with Altman, resigned from the board in a move that OpenAI said would eliminate a “potential future
conflict for Elon” due to Tesla’s work on building self-driving systems.

While OpenAI’s board has preserved its nonprofit governance structure, the startup it oversees has increasingly sought to capitalize on its technology by
tailoring its popular chatbot to business customers.

At its first developer conference last week, Altman was the main speaker showcasing a vision for a future of AI agents that could help people with a variety of
tasks. Days later, he announced the company would have to pause new subscriptions to its premium version of ChatGPT because it had exceeded capacity.

Altman’s exit “is indeed shocking as he has been the face of” generative AI technology, said Gartner analyst Arun Chandrasekaran.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, Elon Musk, Fortune, OpenAI, firing Altman, OpenAI board, OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are
shocked and saddened by what the board did', self-driving systems, AI agents, first developer conference, Musk's resignation from the board, pause of new
subscriptions to ChatGPT's premium version, Gartner, Arun Chandrasekaran, Altman's exit
</item 8>

<item 9>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
But years before OpenAI’s near meltdown, there was a little-publicised but ferocious competition in Silicon Valley for control of the technology that is now
quickly reshaping the world, from how children are taught to how wars are fought. The birth of DeepMind Five years before the Napa Valley party and two before
the cat breakthrough on YouTube, Demis Hassabis, a 34-year-old neuroscientist, walked into a cocktail party at Thiel’s San Francisco town house and realised he
had hit pay dirt. There in Thiel’s living room, overlooking the city’s Palace of Fine Arts and a swan pond, was a chessboard. Hassabis had once been the
second-best player in the world in the under-14 category. “I was preparing for that meeting for a year,” Hassabis said. “I thought that would be my unique hook
in: I knew that he loved chess.”

Graph facts:
- Demis Hassabis -PARTICIPATED_IN-> Artificial Intelligence Safety Summit
- Demis Hassabis -LOCATED_IN-> Britain
- Thiel -WORKS_FOR-> PayPal
- Thiel -PARTICIPATED_IN-> conference
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> disinformation

Entities: Napa Valley, Thiel, Palace of Fine Arts, chess, Demis Hassabis, How ego and fear fuelled the rise of artificial intelligence, DeepMind, The Age,
Silicon Valley, artificial intelligence, San Francisco, OpenAI, YouTube
</item 9>

<item 10>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
“This is another indication that AI is progressing faster than even many experts anticipated,” Victoria Krakovna, who would soon join DeepMind as an “AI safety”
researcher, wrote in a blog post. DeepMind’s founders were increasingly worried about what Google would do with their inventions. In 2017, they tried to break
away from the company. Google responded by increasing the salaries and stock award packages of the DeepMind founders and their staff. They stayed put. The
ethics board never had a second meeting. The Breakup

Convinced that Page’s optimistic view of AI was dead wrong, and angry at his loss of DeepMind, Musk built his own lab. OpenAI was founded in late 2015, just a
few months after he met with Altman at the Rosewood hotel in Silicon Valley. Sam Altman Credit: Bloomberg Businessweek Musk pumped money into the lab, and his
former PayPal buddies – Hoffman and Thiel – came along for the ride.

Graph facts:
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> Musk
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> Musk
- Founders, are events useful? -MENTIONS-> Musk
- X (formerly Twitter) is worst for disinformation, per EU analysis -MENTIONS-> Musk
- The UK just laid out new rules for the internet — it only gets harder from here -MENTIONS-> Musk

Entities: Musk, Page, Rosewood hotel, Thiel, How ego and fear fuelled the rise of artificial intelligence, Victoria Krakovna, Bloomberg Businessweek, Hoffman,
PayPal, ethics board, break away from Google, DeepMind, The Age, Google, Silicon Valley
</item 10>
=== END ITEM id=fc04bfae5fcc71ad ===
