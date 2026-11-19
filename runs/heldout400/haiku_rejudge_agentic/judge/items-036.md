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

=== ITEM id=0bfa1c9c6810acbb ===
Gold sentences and facts:
Sentence 0: The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.
  Fact 0.0: The agreement also meant that Apple never switched to a Google competitor.
  Fact 0.1: The agreement also meant that Apple never allowed users to choose their browser when setting up their iPhones.
Sentence 1: Apple has already indicated that it plans to appeal, but appealing is a lengthy process that can take around 18 months.
  Fact 1.0: Apple has already indicated that it plans to appeal.
  Fact 1.1: Appealing is a lengthy process.
  Fact 1.2: Appealing can take around 18 months.
Sentence 2: They use their store, their payments, they force developers to all have the same terms, they force OEMs and carriers to all have the same terms.
  Fact 2.0: The party described uses its own store.
  Fact 2.1: The party described uses its own payments.
  Fact 2.2: The party described forces developers to all have the same terms.
  Fact 2.3: The party described forces OEMs to all have the same terms.
  Fact 2.4: The party described forces carriers to all have the same terms.

Retrieved items:
<item 1>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
In 2013, then President Barack Obama vetoed an import ban on the iPhone. At the time, Samsung accused Apple of infringing on its cellular data patents. But in
that case, according to Levi, Apple was able to make the argument that there were issues regarding fair, reasonable, and nondiscriminatory licensing (FRAND).
The patents involved were considered standard and essential, and the Obama administration was concerned the import ban would give Samsung “undue leverage.”
Plus, it only impacted the iPhone 4 and older models of the iPad. Newer iPhones didn’t use an infringing chipset. In the case of the Apple Watch, Apple doesn’t
have any of these arguments.

“In that case 10 years ago, Apple really rolled the dice and the pulled the rabbit out of the hat. It’s very unlikely for them to do that again,” says Levi.

Where does Apple go from here?

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, iPad, Why the Apple Watch is being banned — and how Apple can avoid it, Barack Obama, veto of an import ban on the iPhone, Samsung, Cellular
data patents, Levi, Fair, reasonable, and nondiscriminatory licensing (FRAND), Obama administration, Apple Watch, iPhone, The Verge
</item 1>

<item 2>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android

Entities: Android, Apple, Tim Sweeney on Epic’s victory royale over Google, Google case, Apple case, antitrust trickery, restrictions on competition, Tim
Sweeney, Epic, Google, The Verge
</item 2>

<item 3>
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
</item 3>

<item 4>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Cue -PARTICIPATED_IN-> Cue’s testimony
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Bellshaw, ISA, Google Search deal, Google’s privacy policies,
Apple’s financial filings, Google Search, Apple, DOJ, Google, user privacy, Safari, The Verge
</item 4>

<item 5>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh
antitrust scrutiny as Google.

It’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in
the open, I should say — certainly they still wanted to keep them secret.

But I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they
were similarly self-serving in terms of deals.

I’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was
awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners,
you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent.

Graph facts:
- Microsoft -MENTIONS-> device-agnostic ecosystem
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Microsoft
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Microsoft
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Microsoft
- Microsoft -MENTIONS-> cloud gaming

Entities: Microsoft, Apple, Tim Sweeney on Epic’s victory royale over Google, Android operating system, antitrust trial against Microsoft, antitrust scrutiny,
Tim Sweeney, Epic, Google, iOS, Epic v. Apple trial, The Verge
</item 5>

<item 6>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
“From a legal perspective, whatever change Apple makes has to be significant enough to avoid the patent, and it cannot be what’s called a ‘colorable’
difference. Minor changes will not be sufficient,” Iancu says.

Basically, Apple can’t simply make a small tweak to the code and call it a day. Masimo claims that the patents are hardware related, so even completely
disabling the sensor may not work depending on how the patent is written. Apple could remove it entirely from the design, but that would require mucking around
with Apple’s meticulous supply chain and could take some time.

Then there are the options that would cost Apple some money. Apple could sit down with Masimo and hammer out a licensing deal for the infringed patents. In a
New York Times interview, Masimo CEO Joe Kiani said Apple “had not engaged in licensing negotiations.” It’s an easy-seeming fix, so why not do it? For starters,
both parties have to want to settle.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, Why the Apple Watch is being banned — and how Apple can avoid it, Masimo, Apple Watch, Joe Kiani, patents, Iancu, New York Times, The Verge
</item 6>

<item 7>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
For starters, both parties have to want to settle.

“Apple is a tough nut to crack,” says Brittingham, noting that Apple is one of the most sued companies in Silicon Valley. The company’s large cash reserves also
make taking Apple to court a costly endeavor. Masimo, for example, has reportedly spent $60 million litigating the issue thus far. “I think they tend to hold
out as long as possible in order to make it less appetizing to sue Apple.”

The other option is for Apple to avoid the import ban altogether. Iancu says Apple could simply decide to manufacture the watches within the US instead of
overseas — but that’s a significant supply chain change that’s unlikely to happen any time soon.

Does this set a precedent?

This import ban isn’t the only one Apple is facing. In February, the Biden administration also declined to veto another ITC import ban and cease-and-desist
order stemming from a dispute with AliveCor over the Apple Watch’s EKG features.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, US, Why the Apple Watch is being banned — and how Apple can avoid it, Masimo, Apple Watch, Brittingham, Iancu, ITC, EKG features, Apple Watch
import ban, ITC import ban and cease-and-desist order, Dispute with AliveCor over Apple Watch EKG features, Litigating the issue, AliveCor, Silicon Valley
</item 7>

<item 8>
Why the Apple Watch is being banned — and how Apple can avoid it (The Verge, 2023-12-19T18:16:57+00:00)
And after the 24th, they’ll disappear from Apple Stores, too.

“I think Apple sees the writing on the wall and they’re preparing themselves,” says Ben Levi, a partner at Levi Snotherly & Schaumberg, which has experience
litigating ITC disputes.

It’s rare to see Big Tech lose — and with such tangible consequences at that. There are still ways that Apple can keep the watches on sale, but it’ll likely
take a lot of waiting or a lot of money to make it happen.

How Apple got here

This ban is the result of a long-standing patent dispute between Apple and the medical device maker Masimo. The latter is known for its pulse oximetry tech,
generally referred to as SpO2 or blood oxygen saturation in the wearable world, and it claims that Apple is using that patented technology without permission.

This particular story started about 10 years ago when Apple reached out to Masimo about a potential partnership around blood oxygen features on its wearables.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, Ben Levi, Levi Snotherly & Schaumberg, pulse oximetry tech, patent dispute between Apple and Masimo, using patented technology without
permission, Apple Stores, Why the Apple Watch is being banned — and how Apple can avoid it, Masimo, Apple Watch, ITC, The Verge
</item 8>

<item 9>
Graph query row:
article: Tim Sweeney on Epic’s victory royale over Google
published_on: 2023-12-12T19:17:14+00:00
publisher: The Verge
focused_connections: [{'connected_entity': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'entity': 'Google
Play', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'What to know about Fortnite maker Epic
Games’ antitrust battle with Google, starting today', 'entity': 'Google Play', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship':
'MENTIONS'}, {'connected_entity': 'ChatGPT: Everything you need to know about the AI-powered chatbot', 'entity': 'Google Play', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity':
'Google Play', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The Epic v. Google trial may come
down to simple v. complicated', 'entity': 'Google Play', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Beeper Mini is back in operation after Apple’s attempt to shut it down', 'entity': 'Google Play', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The best comedies you’ll find on streaming right now', 'entity': 'Google Play',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Google', 'entity': 'Google Play',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MADE'}, {'connected_entity': 'Google', 'entity': 'Google Play', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'OWNS'}, {'connected_entity': 'App Store', 'entity': 'Google Play', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'COMPETES_WITH'}, {'connected_entity': 'Android', 'entity': 'Google Play', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'PART_OF'}, {'connected_entity': 'Google apps', 'entity': 'Google Play', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'PART_OF'}, {'connected_entity': 'What to know about Fortnite maker Epic Games’ antitrust battle
with Google, starting today', 'entity': 'Google Play Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'},
{'connected_entity': 'European football: Eintracht Frankfurt hit Bayern for five in famous thrashing', 'entity': 'Google Play Store', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity':
'Google Play Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Best Golf Betting Sites and
Apps - Top Sportsbooks for Golf 2023', 'entity': 'Google Play Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'},
{'connected_entity': 'The Epic v. Google trial may come down to simple v. complicated', 'entity': 'Google Play Store', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'European roundup: Bellingham helps Real Madrid go top while Bayern draw', 'entity':
'Google Play Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Epic v. Google, explained',
'entity': 'Google Play Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'West Indies
clinch T20 series after Shai Hope edges hosts past England total', 'entity': 'Google Play Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'],
'relationship': 'MENTIONS'}, {'connected_entity': 'Google', 'entity': 'Google Play Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'],
'relationship': 'OWNS'}, {'connected_entity': 'Galaxy Store', 'entity': 'Google Play Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'],
'relationship': 'COMPETES_WITH'}, {'connected_entity': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'entity':
'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'News publisher files class
action antitrust suit against Google, citing AI’s harms to their bottom line', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today',
'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'A comprehensive
list of 2023 tech layoffs', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Video games in 2023: Acquisitions, layoffs, unions', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Epic Games', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The Epic v. Google trial may come down to simple v.
complicated', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Epic
v. Google, explained', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Can anyone survive Fortnite as a job?', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': 'App store competition', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'deal between Google and Samsung', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Sideloading', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Off-the-record chats', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Play Store access and payment', 'entity': 'Epic Games', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Google', 'entity': 'Epic Games', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Tim Sweeney', 'entity': 'Epic Games', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Google trial with Epic Games', 'entity': 'Epic Games',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic Games lawsuit against Google
over app store antitrust issues', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'},
{'connected_entity': 'Epic Games’ antitrust battle with Google', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'agreement with OnePlus', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': "Epic's antitrust lawsuit", 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'antitrust battle with Google', 'entity': 'Epic Games', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'antitrust battle with Apple', 'entity': 'Epic Games',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic sued Google', 'entity': 'Epic
Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v. Google trial',
'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v.
Apple trial', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity':
'Epic Games Store', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity':
'Fortnite', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity':
'Mediatonic', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'OWNS'}, {'connected_entity':
'Fortnite', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Tim
Sweeney', 'entity': 'Epic Games', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'FOUNDED'}, {'connected_entity': 'ASX set to
drop as Wall Street’s September slump deepens', 'entity': 'antitrust', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'},
{'connected_entity': "Amazon sellers sound off on the FTC's 'long-overdue' antitrust case", 'entity': 'antitrust', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’',
'entity': 'antitrust', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': 'More from the US v Google
trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'entity': 'antitrust', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'],
'relationship': 'MENTIONS'}, {'connected_entity': 'News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line',
'entity': 'antitrust', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': 'What to know about Fortnite
maker Epic Games’ antitrust battle with Google, starting today', 'entity': 'antitrust', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'],
'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'antitrust', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': '5 things we learned so far about the Google antitrust case', 'entity': 'antitrust',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Earnings week ahead: Amazon, Microsoft, Alphabet,
Meta, Ford, GM and more', 'entity': 'antitrust', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity':
'Google', 'entity': 'antitrust', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'Amazon',
'entity': 'antitrust', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'major antitrust trial',
'entity': 'antitrust', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'PART_OF'}, {'connected_entity': 'US v. Microsoft antitrust
trial', 'entity': 'antitrust', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'PART_OF'}, {'connected_entity': 'A comprehensive list
of 2023 tech layoffs', 'entity': 'Bandcamp', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity':
'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Bandcamp', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Bandcamp staff layoffs', 'entity': 'Bandcamp', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Songtradr', 'entity': 'Bandcamp', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'OWNS'}, {'connected_entity': 'Epic', 'entity': 'Bandcamp', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'OWNS'}, {'connected_entity': "Apple's policy for third-party apps on its devices", 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': "200+ of the best deals from Amazon's Cyber Monday sale", 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'ASX set to drop as Wall Street’s September slump deepens',
'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Apple defends Google
Search deal in court: ‘There wasn’t a valid alternative’', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo', 'entity': 'Apple',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'What to know about Fortnite maker Epic
Games’ antitrust battle with Google, starting today', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Founders, are events useful?', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': 'One year later, ChatGPT is still alive and kicking', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'ChatGPT: Everything you need to know about the AI-powered chatbot', 'entity':
'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'A comprehensive list of 2023 tech
layoffs', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The best Black
Friday deals you can get for under $25', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The best Apple Black Friday deals on iPads, AirPods, Apple
Watches and more', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Apple
bets on generative AI', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity':
'OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': "Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making
me laugh'", 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Meta seeks
legislation that would require parents to approve teens’ app downloads', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'NASCAR Might Have Just Saved Themselves With the $7.7 Billion Media Rights
Deal', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'If your startup
doesn’t seem impossible at first, it’s not hard enough', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': '5 things we learned so far about the Google antitrust case',
'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Why the Apple Watch is
being banned — and how Apple can avoid it', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Google', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'The 30 best Amazon Prime Day deals for October 2023', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': "What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs", 'entity': 'Apple',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The Epic v. Google trial may come down to
simple v. complicated', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity':
'Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'These Are Our Favorite Walmart Cyber Monday Deals', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Google fakes an AI demo, Grand Theft Auto VI goes viral and
Spotify cuts jobs', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity':
'Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The best deals on MacBooks right now', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Epic v. Google, explained', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'ASX to rise after interest rate hopes boost US shares',
'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': '8 predictions for AI in
2024', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Here are the best
deals on headphones and wireless earbuds we could find', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Temu was the most-downloaded iPhone app in the U.S. in 2023', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Beeper Mini is back in operation after Apple’s attempt to shut it down',
'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Know the price-matching
policies for Best Buy, Target, Walmart, and others', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The Verge’s 2023 holiday gift guide for kids', 'entity':
'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Warner Music Group boss Robert
Kyncl on AI, why labels still exist, and 2 other things we learned from his Q&A at the Code Conference', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'French soccer players back African mobile video network StarNews Mobile in $3M
round', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'EU commissioner
sidesteps MEPs’ questions about CSAM microtargeting', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'ASX set to rise ahead of inflation report after big tech lifts Wall Street', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The Apple Watch’s double tap gesture points at a new way to
use wearables', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Cell
phone shocker as 97% of kids use their device during school hours and beyond, says study', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The best portable chargers for your gear', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': '78 Absolute Best October Prime Day Deals (2023)', 'entity':
'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Apple 16-inch M3 Max MacBook Pro
review: A desktop among laptops', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'CEO David Baszucki’s mission to make Roblox a billion-player platform', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Beeper reversed engineered iMessage to bring blue bubble texts to Android
users', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Shop 200+
Walmart doorbuster deals for Black Friday', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'The 26 best Amazon Prime Day deals you can still shop for today', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Some of our favorite devices are on sale for Black Friday.', 'entity': 'Apple',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': '54 Best Apple Black Friday Deals (2023):
iPad, Apple Watch, AirPods', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'The best anti-Prime Big Deal Day deals', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': 'The best Prime Day deals on headphones for October 2023', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The best comedies you’ll find on streaming right now',
'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Palestine’s growing tech
industry has been literally blown apart by Israel’s war on Hamas', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': '41 Best Prime Day Apple Deals (2023)', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'ChatGPT', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'M3 chips', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'import ban', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'SAID'}, {'connected_entity': 'MacBook Pro', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'end-to-end encryption', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Assistive Touch', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'S9 SiP', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Dynamic Caching', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Beeper Mini', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Amit Mehta', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Cue', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity':
'Rosenberg', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Ron', 'entity':
'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Sacha Haworth', 'entity': 'Apple',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Tim Sweeney', 'entity': 'Apple',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Pichai', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Masimo', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Samsung', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Levi', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'Brittingham', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'Joe Kiani', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'SAID'}, {'connected_entity': 'Iancu', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'AliveCor', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Google', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Yvonne Gonzalez Rogers', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Elizabeth Warren', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Tim Cook', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'WORKS_FOR'},
{'connected_entity': 'Nikki Rothberg', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'WORKS_FOR'},
{'connected_entity': 'unnamed Apple spokesperson', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'WORKS_FOR'}, {'connected_entity': 'Google-Apple revenue-share deal', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'antitrust battle with Apple', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Bloomberg report on Apple layoffs', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v. Apple', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Apple “Scary Fast” October event', 'entity': 'Apple',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'lucrative search agreement',
'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'DOJ investigation
of Apple', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Apple
Watch removal from Apple’s online store', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PARTICIPATED_IN'}, {'connected_entity': 'patent dispute between Apple and Masimo', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Masimo lawsuit against Apple', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Apple lawsuit against Masimo', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'veto of an import ban on the iPhone', 'entity':
'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'appeal', 'entity': 'Apple',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Dispute with AliveCor over Apple
Watch EKG features', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'},
{'connected_entity': 'Litigating the issue', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PARTICIPATED_IN'}, {'connected_entity': 'Scary Fast', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PARTICIPATED_IN'}, {'connected_entity': 'WWDC', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PARTICIPATED_IN'}, {'connected_entity': 'Epic v. Apple trial', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Apple trial', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v. Apple case', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'favoring its own products', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'patent infringement', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'using patented technology without permission', 'entity': 'Apple',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'making an Apple Watch clone', 'entity':
'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'Cellular data patents',
'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'Cupertino', 'entity':
'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'LOCATED_IN'}, {'connected_entity': 'Palestine', 'entity': 'Apple',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'LOCATED_IN'}, {'connected_entity': 'AI tools', 'entity': 'Apple',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Apple GPT', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'AirPods Pro', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'M3 MacBook Pro', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'M3-enabled Air models', 'entity': 'Apple', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Mac mini', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'AirPods', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'AirPods Max', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MADE'}, {'connected_entity': '16-inch M3 Max MacBook Pro', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'M3 iMac 24-inch', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'iTunes', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MADE'}, {'connected_entity': 'iPad', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MADE'}, {'connected_entity': 'Apple iPad (9th Gen)', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MADE'}, {'connected_entity': 'MacBook Pro M3', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'Apple Watch Series 6', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'Apple Watch', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'wearable EKG feature', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'MacBook Pro', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'MacBooks', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'MacBook Air', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'M2 MacBook Air', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': '15-inch M2 Air', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'Beats Fit Pro', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'M-series chips', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'Apple AirPods Pro (2nd Generation)', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MADE'}, {'connected_entity': "Apple's latest entry-level wearable", 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MADE'}, {'connected_entity': 'Apple Watch Series 9', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MADE'}, {'connected_entity': 'Apple Pencil', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MADE'}, {'connected_entity': 'Apple AirPods (2nd Gen)', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MADE'}, {'connected_entity': 'charger from Apple', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MADE'}, {'connected_entity': 'Apple 35-Watt Dual USB-C Port Compact Power Adapter', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'App Store', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Beats', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'OWNS'}, {'connected_entity': 'Apple Arcade', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'OWNS'}, {'connected_entity': 'Pro Display XDR', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'OWNS'}, {'connected_entity': 'M-series chips', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'OWNS'}, {'connected_entity': 'Final Cut Pro', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'OWNS'}, {'connected_entity': 'Photos', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'OWNS'}, {'connected_entity': 'Safari', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'OWNS'},
{'connected_entity': 'Google', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'OpenAI', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'TicWatch Pro 5', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'Apple’s system software product marketing group', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PART_OF'}, {'connected_entity': 'Steve Wozniak', 'entity': 'Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'FOUNDED'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Google Play billing system', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity':
'Google antitrust trial', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney', 'entity':
'Google antitrust trial', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Tim Sweeney on
Epic’s victory royale over Google', 'entity': 'Google’s control over Android apps', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'illegal monopoly over Android app distribution and in-app
payment systems', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory
royale over Google', 'entity': 'tying Google Play billing system to its app store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'major antitrust trial', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney', 'entity': 'major antitrust trial', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic', 'entity': 'major antitrust trial', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'antitrust', 'entity': 'major antitrust trial', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PART_OF'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'legal
process', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Shayda director Noora Niasari on family
violence, Iranian liberation and taking her first film to the Oscars', 'entity': 'legal process', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'],
'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'antitrust issue', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity':
'Google’s internal documents', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Thing'], 'relationship': 'MENTIONS'}, {'connected_entity': 'What to know about
Fortnite maker Epic Games’ antitrust battle with Google, starting today', 'entity': 'Google Play billing', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Google Play billing',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over
Google', 'entity': 'mobile app store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim
Sweeney on Epic’s victory royale over Google', 'entity': 'Activision-Google deal', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Activision', 'entity': 'Activision-Google deal', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship':
'PARTICIPATED_IN'}, {'connected_entity': 'Google', 'entity': 'Activision-Google deal', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Epic’s victory royale over Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Epic', 'entity': 'Epic’s victory royale over
Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Google', 'entity': 'Epic’s victory
royale over Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Tim Sweeney on Epic’s
victory royale over Google', 'entity': 'Google case', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Apple case', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'],
'relationship': 'MENTIONS'}, {'connected_entity': 'Epic v. Google, explained', 'entity': 'Apple case', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'antitrust trickery', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney', 'entity': 'antitrust trickery', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'SAID'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'antitrust
trial against Microsoft', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s
victory royale over Google', 'entity': 'antitrust scrutiny', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Brave web browser', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'settlement talks with Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google',
'entity': 'payments tie', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney', 'entity':
'payments tie', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'relationship': 'SAID'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale
over Google', 'entity': '9to5Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Publisher'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Abner
Li', 'entity': '9to5Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Publisher'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Tim Sweeney on
Epic’s victory royale over Google', 'entity': 'Epic v. Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Epic v. Google, explained', 'entity': 'Epic v. Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Apple', 'entity': 'Epic v. Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship':
'PARTICIPATED_IN'}, {'connected_entity': 'Epic', 'entity': 'Epic v. Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship':
'PARTICIPATED_IN'}, {'connected_entity': 'Australia', 'entity': 'Epic v. Apple', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship':
'LOCATED_IN'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Epic v. Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Epic v. Google, explained', 'entity': 'Epic v. Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Epic', 'entity': 'Epic v. Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Google', 'entity': 'Epic v. Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Australia', 'entity': 'Epic v. Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'LOCATED_IN'}, {'connected_entity': 'A comprehensive list of 2023 tech layoffs', 'entity': 'Epic',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Video games in 2023: Acquisitions, layoffs,
unions', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on
Epic’s victory royale over Google', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'The Epic v. Google trial may come down to simple v. complicated', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Epic v. Google, explained', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Can anyone survive Fortnite as a job?', 'entity': 'Epic', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Fortnite is the latest game to entice players with a portal
to the past', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Aggressive
pursuit model', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Google',
'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Google tax', 'entity':
'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Tim Sweeney', 'entity': 'Epic',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Yvonne Gonzalez Rogers', 'entity': 'Epic',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Tim Sweeney', 'entity': 'Epic',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Natalie Muñoz', 'entity': 'Epic',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Gary Bornstein', 'entity': 'Epic',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'major antitrust trial', 'entity': 'Epic',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic’s victory royale over Google',
'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v. Apple',
'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v. Google',
'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'original lawsuit',
'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'trial', 'entity':
'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v. Google trial',
'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Apple trial',
'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v. Apple
case', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity':
'settlement', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity':
"Epic's trial", 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity':
'Epic v. Google lawsuit', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'},
{'connected_entity': 'Google', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'},
{'connected_entity': 'anticompetitive practices', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'ACCUSED_OF'}, {'connected_entity': "paying game developers and phone makers not to abandon Google's app store", 'entity': 'Epic', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'Fortnite', 'entity': 'Epic', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Bandcamp', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Houseparty', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'virtual swag shop', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Fortnite', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'OWNS'}, {'connected_entity': 'Sony', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'OWNS'}, {'connected_entity': 'Tencent', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'OWNS'},
{'connected_entity': 'Google', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'Coalition for App Fairness', 'entity': 'Epic', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PART_OF'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'app store restrictions', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Topic'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Epic Games Store',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The best controllers for PC gaming', 'entity':
'Epic Games Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The Epic v. Google trial may
come down to simple v. complicated', 'entity': 'Epic Games Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Tim Sweeney', 'entity': 'Epic Games Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'SAID'},
{'connected_entity': 'Epic Games', 'entity': 'Epic Games Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'MADE'},
{'connected_entity': 'Steam', 'entity': 'Epic Games Store', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Product'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'US v. Microsoft antitrust trial', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Microsoft', 'entity': 'US v. Microsoft antitrust trial', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'antitrust', 'entity': 'US v. Microsoft antitrust trial',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PART_OF'}, {'connected_entity': 'Apple', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'AI Principles', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Sam Altman backs teens’ startup, Google unveils the Pixel 8
and TikTok tests an ad-free tier', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'More from the US v Google trial: vertical search, pre-installs and the case of
Firefox/Yahoo', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Why
Mozilla is betting on a decentralized social networking future', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': 'How ego and fear fuelled the rise of artificial intelligence', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Wired opinion article', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Is Google Search better than the rest? And is that fair?',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'News publisher files
class action antitrust suit against Google, citing AI’s harms to their bottom line', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'There’s something going on with AI startups in France', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'What to know about Fortnite maker Epic Games’ antitrust
battle with Google, starting today', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Founders, are events useful?', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Passwords? We don’t need no steenkin’ passwords', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'One year later, ChatGPT is still alive and kicking', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'ChatGPT: Everything you need to know about the AI-powered
chatbot', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'A
comprehensive list of 2023 tech layoffs', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Govee’s colorful LED light strips are on sale just in time for Halloween', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'From Ten Hag’s future to a
leaking roof: Ratcliffe’s Manchester United in-tray', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'The best Black Friday deals you can get for under $25', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Everton players and fans will be fired up by points penalty, asserts Sean Dyche', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Discogs’ vibrant vinyl community
is shattering', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Biden’s
AI EO hailed as broad, but not deep without legislation to match', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': 'Chelsea secure last-eight spot after Raheem Sterling sinks Blackburn', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Why We Desperately Need a New (and Better) Google',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The people who ruined
the internet', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'People
are turning to Snap Map for firsthand perspectives from Gaza', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': 'Early impressions of Google’s Gemini aren’t great', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The restaurant nearest Google', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Meta seeks legislation that would require parents to approve
teens’ app downloads', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity':
'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'England thrash Scotland but cannot avoid Team GB Olympic heartbreak', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': '‘Mass breach of privacy’: TikTok under fire for tracking users online',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'If your startup doesn’t
seem impossible at first, it’s not hard enough', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'The best Prime Day smart home deals', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': 'England choose survival over oblivion and a chance of summer trip to France', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The 53 best Black Friday deals we could
find at Amazon, Walmart, Target and more', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': "Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court", 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The best VPN services for 2023', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Keep Labs’ smart storage redefines
medication management', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity':
'The best October Prime Day gaming deals', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': '5 things we learned so far about the Google antitrust case', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Ten Hag’s gamble pays off against Chelsea as McTominay steals show | Jonathan Liew',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Deepfake election risks
trigger EU call for more generative AI safeguards', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'The 30 best Amazon Prime Day deals for October 2023', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Meta and IBM form an AI Alliance, but to what end?', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'NMPA: Generative AI is ‘the greatest risk to the human
creative class that has ever existed’', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'The Epic v. Google trial may come down to simple v. complicated', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Deft is building e-commerce search to help you find the right product quickly', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': "Sweeping White House AI executive
order takes aim at the technology's toughest challenges", 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'These Are Our Favorite Walmart Cyber Monday Deals', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Ten Hag demands both positivity and pragmatism from erratic Onana', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Google fakes an AI demo, Grand Theft Auto
VI goes viral and Spotify cuts jobs', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'European roundup: Bellingham helps Real Madrid go top while Bayern draw',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Epic v. Google,
explained', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The Ray-Ban
Meta smart glasses actually make the future look cool', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': '8 predictions for AI in 2024', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': 'Here are the best deals on headphones and wireless earbuds we could find', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Ten Hag’s Manchester United stand up and
die at club stuck in the past tense', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'Sony Music has issued nearly 10,000 deepfake takedowns… and other things we learned from Dennis Kooker’s speech about AI', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Sony Music’s artists aren’t
involved in YouTube’s new voice-cloning AI experiment', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Sony Music’s artists aren’t involved in YouTube’s new voice-cloning AI experiment.', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Sony Music’s artists aren’t involved in YouTube’s new
voice-cloning AI experiment. Not unrelated: Google’s recent filing with the US Copyright Office.', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Temu was the most-downloaded iPhone app in the U.S. in 2023', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Know the price-matching policies
for Best Buy, Target, Walmart, and others', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Israel’s startup ecosystem: down but not out', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Warner Music Group boss Robert Kyncl on AI,
why labels still exist, and 2 other things we learned from his Q&A at the Code Conference', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'French soccer players back African mobile video network StarNews Mobile in $3M round',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'EU commissioner
sidesteps MEPs’ questions about CSAM microtargeting', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'The new, ‘efficient’ Spotify has a very different approach to podcasting', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'ASX set to rise ahead of inflation report after big tech
lifts Wall Street', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity':
'Spill toasts one year with a 2.7M seed extension, Kerry Washington and champagne', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The UK just laid out new rules for the internet — it only gets harder from here', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Tesla ‘digs its own grave with
the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'MENTIONS'}, {'connected_entity': '78 Absolute Best October Prime Day Deals (2023)', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'EU says incoming rules for general purpose AIs can evolve over time',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': "India to be one of
world's fastest growing economies over next decade", 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Liverpool enjoy home comforts but Dyche’s Everton offer
derby danger | Andy Hunter', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'},
{'connected_entity': 'The 26 best Amazon Prime Day deals you can still shop for today', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Some of our favorite devices are on sale for Black Friday.', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The best anti-Prime Big Deal Day deals',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Pebble, the Twitter
alternative previously known as T2, is shutting down', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'The best Prime Day deals on headphones for October 2023', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Palestine’s growing tech industry has been literally blown apart by Israel’s
war on Hamas', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The best
apps and tools for managing your money online', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'MENTIONS'}, {'connected_entity': 'Is Google Search better than the rest? And is that fair?', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'App store competition', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Chat logs and documents provided to the court', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'workforce reductions', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'open source AI', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Apple', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'misinformation', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Bard', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Gemini Pro', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Thai food near me', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'local search', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'bias', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'Activision competing store', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Epic', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'ethical reviews', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'security', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'SAID'}, {'connected_entity': 'safety', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Meta’s open source AI approach', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'SAID'}, {'connected_entity': 'AI training', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Android sideloading flow', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'AI-generated content', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Gemini', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'sideloaded apps equal gaping security holes', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'fair use', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'SAID'}, {'connected_entity': 'YouTube', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'Artificial Intelligence and Copyright', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'ingestion of copyrighted material for training generative AI platforms', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'US copyright law', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'copyright harvesting', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'privacy', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'generative AI', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Cue', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Rosenberg', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Barry Diller', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Mitchell Baker', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Suleyman', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'Megan Grey', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'Helena World Chronicle', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'Epic Games', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'federal government', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'SAID'}, {'connected_entity': 'Forrester', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'SAID'}, {'connected_entity': 'Sullivan', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'},
{'connected_entity': 'SEOs', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity':
'Ray', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Cutts', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Lee', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Sacha Haworth', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Tim Sweeney', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Don Harrison', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Samantha Hoffman', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Wilkins', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'U.S. government', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Pichai', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'DOJ', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Gary Bornstein', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Bornstein', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Pomerantz', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Zach Hudson', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Justice Department', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'SAID'}, {'connected_entity': 'Gary Wang', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'John Schmidtlein', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Jamie Rosenberg', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Sundar Pichai', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Hinton', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Alan Eustace', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Kent Walker', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Matt Cutts', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Danny Sullivan', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Sullivan', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Cutts', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Prabhakar Raghavan', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'David Drummond', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Glenn Pomerantz', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Ilya Sutskever', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Lewis', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Bradley Horowitz', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Dalia Awad', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'WORKS_FOR'}, {'connected_entity': 'Google-Apple revenue-share deal', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Google antitrust lawsuit with Match Group', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Google trial with Epic Games',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'US v Google
trial', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity':
'auction', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity':
'quality tests', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity':
'Epic Games lawsuit against Google over app store antitrust issues', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic Games’ antitrust battle with Google', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'deal between Google and Samsung', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Google developer agreements', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'three-year agreement', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': "Epic's antitrust lawsuit",
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': "Google's
antitrust lawsuit with the Department of Justice", 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PARTICIPATED_IN'}, {'connected_entity': 'antitrust battle with Google', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': '2023 I/O event', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Activision-Google deal', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic’s victory royale over Google', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v. Google', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'U.S. government’s antitrust trial against Google',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'lucrative search
agreement', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity':
'Talks to secure licenses for artists’ vocals and musical melodies', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'trial', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic sued Google', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v. Google trial', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'tentative settlement over app store antitrust claims', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'settlement', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'US v. Google antitrust
trial', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': "Epic's
trial", 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Epic v.
Google lawsuit', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity':
'Google’s submission to the United States Copyright Office', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Google’s recent filing with the US Copyright Office', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Google filing with the US Copyright Office',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Google USCO
submission', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity':
'Google’s USCO filing', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PARTICIPATED_IN'},
{'connected_entity': 'US v Google trial', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'},
{'connected_entity': 'anticompetitive behavior', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'ACCUSED_OF'}, {'connected_entity': 'U.S. antitrust law', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'ACCUSED_OF'}, {'connected_entity': 'Sherman Act', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'ACCUSED_OF'}, {'connected_entity': 'class action antitrust suit', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'ACCUSED_OF'}, {'connected_entity': 'misappropriation of Publishers’ content', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'Epic Games lawsuit against Google over app store antitrust issues',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'class action
antitrust suit against Google', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'},
{'connected_entity': 'digital ad technologies', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'ACCUSED_OF'}, {'connected_entity': 'anti-competitive practices', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'ACCUSED_OF'}, {'connected_entity': 'search monopoly', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'ACCUSED_OF'}, {'connected_entity': 'antitrust', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'ACCUSED_OF'}, {'connected_entity': 'monopoly', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'ACCUSED_OF'}, {'connected_entity': 'anti-competition law', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'systematically destroying chats', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'anticompetitive conduct', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'antitrust law', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'unlawful monopoly', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'making potential evidence disappear', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'illegal tying', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'Android app distribution', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'Android in-app payment
processing', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'}, {'connected_entity': 'deleting
potentially incriminating messages', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'},
{'connected_entity': 'Epic', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'ACCUSED_OF'},
{'connected_entity': 'US', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'LOCATED_IN'},
{'connected_entity': 'Israel', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'LOCATED_IN'},
{'connected_entity': 'Gmail', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'Pixel 8', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'Chrome', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'},
{'connected_entity': 'Play', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity':
'Google travel listings', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity':
'Google Search', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'User
Choice Billing', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google
first-party billing system', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity':
'Google Play', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Search',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google Pixel Buds Pro',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google Pixel Watch',
'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Bard', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Gemini Pro', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google Maps', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google One VPN', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google-developed AI tools', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'enterprise-focused AI services', 'entity':
'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'multi-search', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'multi-search feature', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Nest Cam', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Gemini', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Gemini Ultra', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google Pixel Buds A-Series', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google Cloud Platform', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google reCaptcha', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google Nest Thermostat', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Google Nest Hub (second-gen)', 'entity': 'Google',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'MADE'}, {'connected_entity': 'Gmail', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Google Search', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Google Play', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Google Play Store', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'DeepMind', 'entity': 'Google', 'entity_labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Bard', 'entity': 'Google', 'entity_labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Riot', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Google Maps', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'relationship': 'OWNS'}, {'connected_entity': 'Gemini', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'relationship': 'OWNS'}, {'connected_entity': 'YouTube', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'OWNS'}, {'connected_entity': 'Alphabet', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'OWNS'},
{'connected_entity': 'Yahoo', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'DuckDuckGo', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'ChatGPT', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'Bing', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'Apple', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'Reddit', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'COMPETES_WITH'},
{'connected_entity': 'large language models', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'COMPETES_WITH'}, {'connected_entity': 'Brave', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'COMPETES_WITH'}, {'connected_entity': 'OpenAI', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'COMPETES_WITH'}, {'connected_entity': 'TikTok', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'COMPETES_WITH'}, {'connected_entity': 'Epic', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'COMPETES_WITH'}, {'connected_entity': 'IBM', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'COMPETES_WITH'}, {'connected_entity': 'Deft', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'COMPETES_WITH'}, {'connected_entity': 'TicWatch Pro 5', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'COMPETES_WITH'}, {'connected_entity': 'Alphabet', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PART_OF'}, {'connected_entity': 'Partnership on AI', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PART_OF'}, {'connected_entity': 'Alphabet Inc.', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PART_OF'}, {'connected_entity': 'Gradient Ventures', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PART_OF'}, {'connected_entity': 'DeepMind', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PART_OF'},
{'connected_entity': 'Google Users & Products', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship':
'PART_OF'}, {'connected_entity': 'Area 120', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PART_OF'},
{'connected_entity': 'web spam team', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'PART_OF'},
{'connected_entity': 'Larry Page', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'FOUNDED'},
{'connected_entity': 'Sergey Brin', 'entity': 'Google', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'relationship': 'FOUNDED'},
{'connected_entity': 'Tim Sweeney on Epic’s victory royale over Google', 'entity': 'Epic v. Apple trial', 'entity_labels': ['__KGBuilder__', '__Entity__',
'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'The Epic v. Google trial may come down to simple v. complicated', 'entity': 'Epic v. Apple trial',
'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Epic v. Google, explained', 'entity': 'Epic v.
Apple trial', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'MENTIONS'}, {'connected_entity': 'Epic Games', 'entity': 'Epic v.
Apple trial', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Apple', 'entity': 'Epic v.
Apple trial', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Phil Schiller', 'entity':
'Epic v. Apple trial', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Tim Sweeney',
'entity': 'Epic v. Apple trial', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}, {'connected_entity': 'Benjamin
Simon', 'entity': 'Epic v. Apple trial', 'entity_labels': ['__KGBuilder__', '__Entity__', 'Event'], 'relationship': 'PARTICIPATED_IN'}]
</item 9>

<item 10>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Amit Mehta
- Cue -SAID-> search engines
- Cue -SAID-> Apple's own search product
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Eddy Cue
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> US v. Google

Entities: Cue, Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’, Bellshaw, App Tracking Transparency, Cue’s testimony, location
tracking, Mozilla, Google Search, Apple, Maps, data collection, Samsung, Google, search, Justice Department
</item 10>
=== END ITEM id=0bfa1c9c6810acbb ===
